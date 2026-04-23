- AI agents are autonomous systems that can interact with external/internal tools to performa a task
    - Problem solving skills
    - Ability to take actions - use external tools to search database, and execute workflows
- [Some source](http://theneuralmaze.substack.com/p/the-ai-engineering-playbook)
### LLM as the brain

- The agent’s core, enabling understanding, reasoning, and response generation

### Tooling

[AI Agent Tools Landscape](https://www.notion.so/AI-Agent-Tools-Landscape-3175ca1cd3c6800793f4c381742b46d3?pvs=21)

- Connects to external tools for enhanced functionality like retrieving data or automating actions.

### Reasoning / Planning routines

- Executes tasks logically with a structured decision-making
    - CoT- Chain of Thoughts
    - CoT-SC - Chain of Thoughts Self Consistency
    - ToT - Three of Thoughts
    - ReAct - Reason and Act.
        - Most famous Reasoning technique
        - Consist of a loop of `Thought-Action-Observation`

### Memory Components

- Retains temporary task data(short-term) and persistent knowledge (long-term) for adaptability
- Short-term
    - Component necessary for application with conversational interface
    - It allow access to a window of previous messages. What can think of it as short-term memory.
    - Special caution must be taking with the size of the window to avoid exceeding the context limit of the LLM
- Long-term
    - like RAG - Chunking and embedding

### Agentic Design Patterns

- Reflection Pattern
    - It allows the LLM to reflect on its results, suggesting modifications, additions, improvements in the writing style, etc.
- Tool Use Pattern
    - The information stored in the LLM weights  is not enough to give accurate and insightful answers to our question. Tools provide the way to access the resources that LLM can use to get the right results.
- Planning Pattern
    - Allow agent to decide what sequence of steps to follow to accomplish a large task. Example: ReAct technique.
- MultiAgent Pattern
    - Tasks are divided into sub-tasks and given to the agents based a their role. CrewAI, AutoGen implement variations of multi-agent pattern.

![image.png](../ai-agents/agent-pattern.webp)