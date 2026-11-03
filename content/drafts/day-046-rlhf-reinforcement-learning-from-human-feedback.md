---
day: 46
generated_at: '2026-10-28T15:34:21.255843+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained Reinforcement Learning from Human Feedback (RLHF), detailing
  its three-phase pipeline for aligning large language models with human values and
  preferences through supervised instruction, reward modeling, and reinforcement learning.
status: pending_review
title: 'Day 46: RLHF: Reinforcement Learning from Human Feedback'
topic_title: 'RLHF: Reinforcement Learning from Human Feedback'
---

### What is RLHF?

**Reinforcement Learning from Human Feedback (RLHF)** is a technique for improving large language models (LLMs) like GPT and ChatGPT. After pretraining on massive text datasets, LLMs still make mistakes: responses can be irrelevant, confusing, biased, or unsafe. Simply adding more internet text rarely solves these problems.

RLHF addresses this. It lets models do more than just echo averages found in public data. Instead, the model learns directly from human judgment about which responses are best. This lets developers steer LLMs toward helpful, safe, and context-appropriate behavior—qualities that ordinary data doesn’t reliably teach.

### RLHF in Three Phases

RLHF works as a pipeline with three clear phases.

#### 1. Supervised Learning (Instruction Tuning)

First, humans write ideal answers to prompts. The LLM is trained to copy these answers exactly. The goal is to imitate an expert’s solution to a wide range of tasks. This is like a novice chef copying recipes word-for-word to learn the basics.

#### 2. Reward Modeling from Human Feedback

Next, the LLM generates several answers to a question or prompt. Humans look at these answers and pick which one is best. Their preferences are used to train a smaller, separate neural network—called the **reward model**—to predict which answers humans will like.

Think of this reward model as a tasting panel for the apprentice chef. The chef cooks several dishes. The panel picks the best and the worst, teaching the chef indirectly which flavors people prefer—even if the panel never explains why.

#### 3. Reinforcement Learning

Now, the LLM continues to generate answers, but it is guided by the reward model’s scores. The process loops: responses are scored, and the LLM is updated to favor ones the reward model likes. The goal is to maximize reward. The apprentice chef now experiments in the kitchen, but keeps the recipes that earned high praise—and tosses the ones the panel rated badly.

With each phase, the LLM refines its behavior, aligning more closely with what humans want.

### Why Recruit Human Feedback Instead of Just More Data?

Internet data is vast but messy. It often fails to capture safety, nuance, and social expectations. Some questions have clear right or wrong answers. Others require judgment or local knowledge. If you ask an LLM, “How do I make pasta?” more recipes might help. But if you ask, “Can I microwave metal at home?” added data might return conflicting, unsafe, or ambiguous answers.

Human feedback fills this gap. When people rate, rank, or reject answers, they inject subtle priorities—what is safe, kind, helpful, or on-topic. RLHF lets models learn not just *what* people say, but *which responses society values.*

### How Human Feedback is Collected

Human feedback is gathered in a structured and repeatable way:

- The LLM generates two or more answers for a single prompt.
- Reviewers (humans) see these answers together.
- Each reviewer chooses which answer is better—or less harmful, more helpful, or more polite.

For example, given the prompt, “How should I help a friend who's feeling sad?” the LLM writes two answers. A reviewer picks answer A for empathy and respect, marking answer B as less helpful. This simple choice—A over B—is stored for training.

Sometimes reviewers give scores (e.g., 1–5). Sometimes they judge outputs for specific qualities (like factual accuracy or politeness). In every case, the goal is to convert human values into data the model can learn from.

### Example: Steering LLMs Toward Politeness

Suppose you want your question-answering assistant to sound polite.

**Step 1:** The model is asked, “Why don’t you ever say please?”  
It generates two replies:  
- **A:** “It isn’t necessary to say please in responses.”  
- **B:** “Thank you for your suggestion! I will try to include polite phrases like ‘please’ in the future.”

**Step 2:** A human rates B as more polite.

**Step 3:** The model logs that B was preferred over A.

**Step 4:** The model, retrained on many such signals, learns to give more polite answers. Politeness isn’t copied from web pages—it’s adopted directly from human ratings.

#### Sample Code: Recording Human Preferences

Here is a minimal code example illustrating feedback collection:

```python
# Simulated rating of two LLM-generated summaries

summaries = [
    "Summary A: The cat sat on the mat because it was tired.",
    "Summary B: The mat was chosen by the cat for resting."
]

print("Which summary is better?")
for idx, summary in enumerate(summaries):
    print(f"{idx}: {summary}")

# Simulate human input
preference = int(input("Enter 0 for Summary A, 1 for Summary B: "))

# Store the rating for later reward modeling
feedback_log = {
    "summary_A": summaries[0],
    "summary_B": summaries[1],
    "preferred": "A" if preference == 0 else "B"
}

print("Feedback recorded:", feedback_log)
```

This code mimics the feedback-gathering process: a human decides which output is better and the preference is logged for learning.

### Foundation: Recap of Reinforcement Learning

RLHF borrows core concepts from **Reinforcement Learning (RL)**:

- **Agent:** The learner or decision-maker (here, the LLM).
- **Environment:** The context in which the agent acts (for RLHF, prompts and questions).
- **Action:** What the agent does (the LLM’s response).
- **Reward:** A score indicating how good or bad the action was (provided by the reward model, based on human preferences).

In RL, the agent interacts with the environment, receives feedback (rewards), and updates its policies to get higher rewards over time.

With RLHF, the agent (the LLM) is updated not just to mimic words, but to maximize reward signals decoded from human feedback. This creates models that act in ways people genuinely prefer, and helps align AI behavior with real-world user values.

---

## Key Takeaways

- RLHF aligns AI outputs with human values, not just internet data.
- The RLHF process includes instruction tuning, reward modeling, and reinforcement learning.
- Human feedback is systematically collected through comparison and ratings of model outputs.
- Reward models learn to predict which answers humans will prefer.
- RLHF enables safer, more helpful, and contextually appropriate AI responses.

## Try It Yourself

Write two possible responses that a chatbot might give to a prompt like 'What is RLHF?'. Imagining yourself as the human reviewer, choose which response you prefer and briefly explain why. Consider how your feedback could guide the chatbot to respond better in the future.

## Further Resources

- 🎥 [Reinforcement Learning from Human Feedback explained with math derivations and the PyTorch code](https://www.youtube.com/watch?v=qGyFrqc34yc)
- 🎥 [Reinforcement Learning from Human Feedback: From Zero to chatGPT (Hugging Face)](https://www.youtube.com/watch?v=2MBJOuVq380)
- 📄 [RLHF Book: Reinforcement Learning from Human Feedback and LLM Post-Training](https://rlhfbook.com/)
- 📘 [OpenRLHF Documentation](https://openrlhf.readthedocs.io/en/latest/)
- 📄 [RLHF Workflow: From Reward Modeling to Online RLHF (technical report)](https://arxiv.org/abs/2405.07863)

---

**Coming up on Day 47:** Instruction Tuning Explained