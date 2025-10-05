# Personal Notes
I can tell by the nature of this assignment that it is meant to test system design and engineering skills over pure coding ability. Especially the ability to forsee potential pitfalls and edge cases in complex AI systems where user input, and AI model outputs, can be unpredictable and adversarial. 

I feel there are still many improvements I'd like to add and improve system robustness, usability, and safety. Of which include the following:

- Interactive Refinement Loop:
  Allow users to be a critic in a feedback loop and tweak stories to their liking using function calling and modular design.
- Debug Logging and Metrics:
  Would be good for future development as complexity increases. Good for also determining bottlenecks from API and elsewhere
- Unit Tests:
  Rather than using ad hoc tests, a structured test set would make it easier to ensure consistency and quickly test edge cases.
- Critique Rubrics:
  Improve the consistency of revisions and critiques and reduce variability in quality (resulting from model temperature).
- Token Cost Logging:
  Gather cost metrics of the system. Especially being that this uses multiple LLM calls.
- LLM Function Calling:
  For clean implementation of modular functions in a more complex system where the user has more control.
- Multilingual Support:
  Historically, this was one of the original purposes of LLMs and it would be nice to incorporate that here.
- Additional Guardrails, Anti-Injection, Malicious Input Detection, etc:
  There's always room for more safety and guardrails as there's no perfect solution.

## How to Run
1. Ensure you have Python 3.12.8 (version used during development)
2. Setup a virtual environment and install dependencies (done using bash terminal)
    ```
    python -m venv venv
    source venv/Scripts/activate
    pip install -r requirements.txt
    ```
3. Add your OpenAI API key to a .env file following the .env-template
4. Run main.py
   ```
   python ./main.py
   ```

## High Level Flow
1. Input cleaning/validation
2. Prompt improvement
3. Outline generation
4. Draft generation
5. Critique and revise
6. Final safety pass

<img alt="image" src="https://github.com/PoyoPoak/hippocratic-ai-assessment/blob/main/high-level-diagram.jpg?raw=true" />
    
## Resources Used
- Miro: Diagramming the system architecture and flow.
- Copilot: For code review, suggestions, documentation writing, issue spotting, quick prototyping, test generation, and boilerplate. 
- OpenAI Docs: For API reference and examples.
- Google: For researching best practices, new design patterns, pitfalls, safety considerations, and other relevant information.
- Old Projects: Having built LLM implementations before, I was able to reuse some code patterns and prompt structures.
- Old AI Class Notes: Drew upon old notes for Actor and Critic design patterns.
- Old Information Retrieval Class Notes: Drew upon old notes for input handling and prompt improvement strategies.

# Hippocratic AI Coding Assignment
Welcome to the [Hippocratic AI](https://www.hippocraticai.com) coding assignment

## Instructions
The attached code is a simple python script skeleton. Your goal is to take any simple bedtime story request and use prompting to tell a story appropriate for ages 5 to 10.
- Incorporate a LLM judge to improve the quality of the story
- Provide a block diagram of the system you create that illustrates the flow of the prompts and the interaction between judge, storyteller, user, and any other components you add
- Do not change the openAI model that is being used. 
- Please use your own openAI key, but do not include it in your final submission.
- Otherwise, you may change any code you like or add any files

---

## Rules
- This assignment is open-ended
- You may use any resources you like with the following restrictions
   - They must be resources that would be available to you if you worked here (so no other humans, no closed AIs, no unlicensed code, etc.)
   - Allowed resources include but not limited to Stack overflow, random blogs, chatGPT et al
   - You have to be able to explain how the code works, even if chatGPT wrote it
- DO NOT PUSH THE API KEY TO GITHUB. OpenAI will automatically delete it

---

## What does "tell a story" mean?
It should be appropriate for ages 5-10. Other than that it's up to you. Here are some ideas to help get the brain-juices flowing!
- Use story arcs to tell better stories
- Allow the user to provide feedback or request changes
- Categorize the request and use a tailored generation strategy for each category

---

## How will I be evaluated
Good question. We want to know the following:
- The efficacy of the system you design to create a good story
- Are you comfortable using and writing a python script
- What kinds of prompting strategies and agent design strategies do you use
- Are the stories your tool creates good?
- Can you understand and deconstruct a problem
- Can you operate in an open-ended environment
- Can you surprise us

---

## Other FAQs
- How long should I spend on this? 
No more than 2-3 hours
- Can I change what the input is? 
Sure
- How long should the story be?
You decide
