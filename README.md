# Feature: Chat Tree

Demo video: [`docs/assets/chat-tree-demo.mp4`](docs/assets/chat-tree-demo.mp4)

This branch is suggested to be used together with [`Reekin/CodexMonitor`](https://github.com/Reekin/CodexMonitor/tree/feat/chattree-integration).

## Development Goal

Support tree-structured context management.

## Behavior

The app stores conversation history as a tree.

Each time the user sends a prompt, create a new node after the current node, then set that new node as the current node.

Whenever history is assembled for an LLM request, include only this node and all of its parent nodes recursively.

- Node content: from the user prompt at this node through the end of that turn, whether interrupted or completed.

In a conversation, users can open the chat tree UI with `/chattree`.

The chat tree UI displays all nodes, and users can use arrow keys to choose a different node as the current node.

Chat tree illustration:

```text
[ ]1. Understand current project state
    [ ]2. Discuss the feature approach and break it into steps
        [ ]3. Request implementation of feature step 1
         |  [ ]4. During agent output, notice unclear requirements and add clarification
         |      [ ]5. During testing, find unexpected behavior and discuss + fix
         |          [ ]6. Discuss test feedback after the fix
         |              [ ]7. Continue fixing newly found bugs
        [ ]8. Request implementation of feature step 2
            [ ]9. During testing, find unexpected behavior and discuss + fix
>            |  [*]12. Keep digging into root cause and analyze the error
            [ ]10. Suddenly realize step 1.5 is also required and discuss approach
                [ ]11. Request implementation of step 1.5

Press ↑ ↓ to select nodes; Press space to mark the node as CURRENT NODE;
```

Notes:

- `[*]` indicates the current node, and it moves as you navigate with `↑` and `↓`.
- Numbering is included only to illustrate real chronological order in conversation and is not shown in the actual UI.
- Example order:

```text
1-2-3-4-5-6-7-(set current node from 7 to 2 in chat tree)-8-9-(set current node from 9 to 8 in chat tree)-10-11-(set current node from 11 to 8 in chat tree)-12
```
