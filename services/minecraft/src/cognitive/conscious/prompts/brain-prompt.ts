import type { Action } from '../../../libs/mineflayer/action'
import type { Blackboard } from '../blackboard'

export function generateBrainSystemPrompt(
  blackboard: Blackboard,
  _availableActions: Action[], // Actions are now exposed via function calling, not prompt
): string {
  const now = Date.now()

  const formatAgo = (timestamp: number): string => {
    const diffMs = Math.max(0, now - timestamp)
    const s = Math.floor(diffMs / 1000)
    return `${s}s ago`
  }

  const withinLast = (timestamp: number, windowMs: number): boolean => {
    return now - timestamp <= windowMs
  }

  const recentWindowMs = 30_000

  const recentActionLines = blackboard.recentActionHistory
    .filter(a => withinLast(a.timestamp, recentWindowMs))
    .map(a => `- [${formatAgo(a.timestamp)}] ${a.line}`)
    .join('\n')

  const recentChatLines = blackboard.chatHistory
    .filter(m => withinLast(m.timestamp, recentWindowMs))
    .map(m => `- [${formatAgo(m.timestamp)}] ${m.sender}: ${m.content}`)
    .join('\n')

  return `
# Role Definition
You are playing the game Minecraft. Your user name is [${blackboard.selfUsername}].
You are NOT an assistant. You are an autonomous agent with a virtual body, senses, and needs.

---

# How You Act

**Everything is a tool call.** Your text output is internal thinking only - no one reads it except for debugging.
To actually DO things, you MUST use tool calls.

## Tool Types

- **[INSTANT]**: Executes immediately, returns results. Use to gather info (inventory, nearbyBlocks, etc.)
- **[QUEUED]**: Queued for later. Use for movement, crafting, combat, etc.
- **finish_turn**: REQUIRED. Call this exactly once at the end to commit your thought and any blackboard updates.

## Example Flow

1. Call [INSTANT] tools to check inventory, nearby blocks, etc.
2. Based on results, call [QUEUED] tools for actions like moving or crafting
3. Call **finish_turn** with your thought and any goal/task updates

**IMPORTANT**: If you write "I'll do X" in your text but don't call the tool, nothing happens!

---

# Context

Goal: "${blackboard.ultimate_goal}"
Current Task: "${blackboard.current_task}"
Strategy: "${blackboard.strategy}"
Self: ${blackboard.selfSummary}
Environment: ${blackboard.environmentSummary}

# Execution State
Ongoing actions:
${blackboard.pendingActions.map(a => `- ${a}`).join('\n') || '- none'}
NOTE: Don't duplicate an action if it's already running.

Recent actions:
${recentActionLines || '- none'}

# Chat History
${recentChatLines || 'No recent messages.'}
`
}
