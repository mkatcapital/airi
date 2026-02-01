import type { Agent } from 'neuri'
import type { Message } from 'neuri/openai'

import type { Mineflayer } from '../../libs/mineflayer'
import type { PlanStep } from '../planning/adapter'

import { agent } from 'neuri'
import { system, user } from 'neuri/openai'

import { BaseLLMHandler } from '../../cognitive/conscious/handler'
import { ActionError } from '../../utils/errors'
import { useLogger } from '../../utils/logger'
import { generateActionSystemPrompt } from './system-prompt'
import { actionsList } from './tools'

// --- Async Action Queue (module-level singleton) ---
// Async actions push to this queue during tool execution.
// Brain drains the queue after LLM generation completes.

interface QueuedAction {
  action: string
  params: Record<string, unknown>
  require_feedback?: boolean
}

const MAX_QUEUED_ACTIONS = 5
let asyncActionQueue: QueuedAction[] = []

export function clearAsyncActionQueue(): void {
  asyncActionQueue = []
}

export function drainAsyncActionQueue(): QueuedAction[] {
  const actions = asyncActionQueue
  asyncActionQueue = []
  return actions
}

// --- Finish Turn Data (module-level singleton) ---
// The finish_turn tool stores its result here for Brain to retrieve.

export interface FinishTurnData {
  thought: string
  ultimate_goal?: string
  current_task?: string
  strategy?: string
}

let finishTurnData: FinishTurnData | null = null
let finishTurnCalled = false

export function clearFinishTurnData(): void {
  finishTurnData = null
  finishTurnCalled = false
}

export function setFinishTurnData(data: FinishTurnData): void {
  finishTurnData = data
}

export function getFinishTurnData(): FinishTurnData | null {
  return finishTurnData
}

function isDuplicateAction(action: string, params: Record<string, unknown>): boolean {
  // Simple deep quality check via JSON stringify
  const paramsStr = JSON.stringify(params)
  return asyncActionQueue.some(
    qa => qa.action === action && JSON.stringify(qa.params) === paramsStr,
  )
}

/**
 * Generate actionable suggestions for common error codes.
 * These help the LLM understand what it can do to recover from failures.
 */
function getSuggestionForError(code: string): string {
  switch (code) {
    case 'RESOURCE_MISSING':
      return 'Suggestion: Check your inventory first, then gather the missing resources or ask the player for help if you cannot find them.'
    case 'TARGET_NOT_FOUND':
      return 'Suggestion: The target could not be found. Ask the player for clarification or use exploration tools to locate it.'
    case 'NO_PATH':
      return 'Suggestion: Unable to reach the destination. Try finding an alternative route or ask the player for guidance.'
    case 'TIMEOUT':
      return 'Suggestion: The action timed out. Consider breaking it into smaller steps or trying again later.'
    default:
      return 'Suggestion: Review the error details and try a different approach, or ask the player for help.'
  }
}

export async function createActionNeuriAgent(mineflayer: Mineflayer): Promise<Agent> {
  const logger = useLogger()
  logger.log('Initializing action agent')
  let actionAgent = agent('action')

  Object.values(actionsList).forEach((action) => {
    const isInstant = action.execution === 'parallel'
    const isFinishTurn = action.name === 'finish_turn'

    actionAgent = actionAgent.tool(
      action.name,
      action.schema,
      async ({ parameters }) => {
        mineflayer.memory.actions.push(action)

        if (isFinishTurn) {
          // Special handling for finish_turn - store data for Brain to retrieve
          const params = parameters as { thought: string, ultimate_goal?: string, current_task?: string, strategy?: string }

          if (finishTurnCalled) {
            logger.warn('[FINISH_TURN] Already called this turn, updating thought')
            // return 'Turn already committed. Updates recorded.' // Optional: could return early
          }

          logger.withFields({ thought: params.thought?.slice(0, 50) }).log('[FINISH_TURN] Committing thought and blackboard')
          setFinishTurnData({
            thought: params.thought,
            ultimate_goal: params.ultimate_goal,
            current_task: params.current_task,
            strategy: params.strategy,
          })
          finishTurnCalled = true
          return 'Turn committed successfully.'
        }
        else if (isInstant) {
          // Instant tools: execute immediately, return result
          logger.withFields({ name: action.name, parameters, type: 'INSTANT' }).log('[INSTANT] Executing tool')
          const fn = action.perform(mineflayer)
          try {
            const result = await fn(...Object.values(parameters))
            logger.withFields({ name: action.name, result: typeof result === 'string' ? result.slice(0, 100) : result }).log('[INSTANT] Tool completed')
            return result
          }
          catch (error) {
            if (error instanceof ActionError) {
              logger.withError(error).warn('[INSTANT] Tool failed with ActionError')
              const contextStr = error.context ? `\nContext: ${JSON.stringify(error.context)}` : ''
              const suggestion = getSuggestionForError(error.code)
              return `[FAILED] ${error.code}: ${error.message}${contextStr}\n${suggestion}`
            }
            throw error
          }
        }
        else {
          // Async tools: queue for later execution
          if (asyncActionQueue.length >= MAX_QUEUED_ACTIONS) {
            logger.warn('[QUEUED] Queue limit reached, action rejected')
            return `[REJECTED] Queue full (${MAX_QUEUED_ACTIONS} actions). Wait for current actions to complete.`
          }

          if (isDuplicateAction(action.name, parameters as Record<string, unknown>)) {
            logger.warn('[QUEUED] Duplicate action rejected')
            return `[REJECTED] Duplicate action ${action.name} already queued.`
          }

          const isChat = action.name === 'chat'
          const defaultFeedback = !isChat
          const requireFeedback = (parameters as any).require_feedback ?? defaultFeedback
          const queuedAction: QueuedAction = {
            action: action.name,
            params: parameters as Record<string, unknown>,
            require_feedback: requireFeedback,
          }
          asyncActionQueue.push(queuedAction)
          logger.withFields({ name: action.name, params: parameters, require_feedback: requireFeedback, queueLength: asyncActionQueue.length }).log('[QUEUED] Action queued for execution')
          return `[QUEUED] ${action.name} with params ${JSON.stringify(parameters)} - will execute after your response completes`
        }
      },
      { description: `${isFinishTurn ? '' : (isInstant ? '[INSTANT] ' : '[QUEUED] ')}${action.description}` },
    )
  })

  return actionAgent.build()
}

export class ActionLLMHandler extends BaseLLMHandler {
  public async executeStep(step: PlanStep): Promise<string> {
    const systemPrompt = generateActionSystemPrompt()
    const userPrompt = this.generateActionUserPrompt(step)
    const messages = [system(systemPrompt), user(userPrompt)]

    const result = await this.handleAction(messages)
    return result
  }

  private generateActionUserPrompt(step: PlanStep): string {
    return `Execute this step: ${step.description}

Suggested tool: ${step.tool}
Params: ${JSON.stringify(step.params)}

Please use the appropriate tool with the correct parameters to accomplish this step.
If the suggested tool is not appropriate, you may choose a different one.`
  }

  public async handleAction(messages: Message[]): Promise<string> {
    const result = await this.config.agent.handleStateless(messages, async (context) => {
      this.logger.log('Processing action...')
      const retryHandler = this.createRetryHandler(
        async ctx => (await this.handleCompletion(ctx, 'action', ctx.messages)).content,
      )
      return await retryHandler(context)
    })

    if (!result) {
      throw new Error('Failed to process action')
    }

    return result
  }
}
