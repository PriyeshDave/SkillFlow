# SkillFlow

**An AI-powered curriculum engine for building continuous, human-reviewed technical learning journeys.**

SkillFlow turns a structured curriculum into an automated content pipeline — generating, researching, critiquing, reviewing, and publishing technical lessons on a recurring schedule.

The first curriculum powered by SkillFlow is:

> **Zero to Agentic: Your 105-Day AI Engineer Roadmap**

---

## 🚀 What is SkillFlow?

Creating a high-quality technical learning series manually is surprisingly difficult.

You need to:

- Design a coherent curriculum
- Research each topic
- Write and explain concepts clearly
- Ensure technical accuracy
- Maintain continuity between lessons
- Create practical examples
- Review and edit every lesson
- Publish consistently
- Track where the learner is in the journey

**SkillFlow automates the entire pipeline while keeping a human in the loop.**

Instead of treating every article as an isolated piece of content, SkillFlow treats the curriculum as a **continuous learning journey**.

```text
                    ┌─────────────────────┐
                    │   Fixed Curriculum  │
                    │   105 Day Roadmap   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Select Next Day   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Generate Outline  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Research & Draft  │
                    │   + Web Fact Check  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   AI Critique       │
                    │   Accuracy + Clarity│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Style & Polish    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Human Review      │
                    │   GitHub PR         │
                    └──────────┬──────────┘
                               │
                          Approved?
                         /         \
                       No           Yes
                       │             │
                       ▼             ▼
                    Revise       ┌──────────────┐
                                 │   Publish    │
                                 │ dev.to +     │
                                 │ LinkedIn     │
                                 └──────┬───────┘
                                        │
                                        ▼
                                Next Curriculum Day
```

---

## 🎯 The Core Idea

SkillFlow is built around one principle:

> **Don't just generate content. Build a learning journey.**

Each lesson understands where the learner has been and where they're going.

Every lesson:

- Recaps the previous day's concept
- Introduces the current concept
- Connects it to the broader curriculum
- Previews the next lesson
- Uses practical examples
- Includes code when appropriate
- Ends with key takeaways
- Provides a hands-on exercise

This makes the series feel like **one continuous course**, rather than a collection of unrelated blog posts.

---

# 📚 Zero to Agentic

The first curriculum built with SkillFlow is a **105-day journey from NLP fundamentals to production-grade AI agents**.

### Curriculum

```text
NLP Foundations
      ↓
Text Representation
      ↓
Embeddings
      ↓
Transformers
      ↓
Large Language Models
      ↓
Prompt Engineering
      ↓
Retrieval-Augmented Generation
      ↓
Fine-Tuning
      ↓
AI Agents
      ↓
Agentic Architectures
      ↓
Tool Use & MCP
      ↓
Multi-Agent Systems
      ↓
Evaluation & Observability
      ↓
Production AI Systems
```

The curriculum is divided into **14 progressive phases**, with each phase building on concepts introduced earlier.

---

# 🧠 How SkillFlow Works

## 1. Curriculum Engine

SkillFlow starts with a predefined curriculum.

Rather than asking an LLM:

> "What should I teach today?"

the system knows exactly where the learner is in the roadmap.

```text
Day 1 → Day 2 → Day 3 → ... → Day 105
```

This provides:

- Deterministic progression
- Consistent coverage
- No duplicated topics
- Controlled learning difficulty
- A predictable learner journey

---

## 2. Multi-Pass Content Generation

A lesson isn't generated in a single LLM call.

SkillFlow uses multiple stages.

### Outline

Defines:

- Learning objective
- Core concepts
- Examples
- Code requirements
- Previous-day connection
- Next-day connection

### Draft

Generates the initial lesson using the outline and supporting research.

### Fact Check

Relevant information is verified using live web research to reduce the risk of outdated or incorrect technical claims.

### Critique

The generated lesson is evaluated for:

- Technical accuracy
- Pedagogical quality
- Clarity
- Logical progression
- Missing explanations
- Unnecessary complexity

### Style Pass

The final pass makes the lesson:

- Clear
- Consistent
- Teaching-oriented
- Easy to scan
- Consistent with the overall series

---

# 🔄 Continuity Engine

One of SkillFlow's most important features is **lesson continuity**.

Traditional AI content generation looks like:

```text
Topic A → Article A

Topic B → Article B

Topic C → Article C
```

SkillFlow instead creates:

```text
Day 1
  ↓
Day 2 ← Recap Day 1
  ↓
Day 3 ← Recap Day 2
  ↓
Day 4 ← Recap Day 3
  ↓
...
Day 105
```

Each lesson contains:

### Previous Day

> What did we learn?

### Today's Lesson

> What are we learning now?

### Next Day

> Why does today's concept matter for what's coming next?

This creates a **connected curriculum narrative**.

---

# 👨‍💻 Human-in-the-Loop

SkillFlow is intentionally **not fully autonomous**.

AI generates the lesson.

**A human approves it.**

Every lesson goes through a GitHub Pull Request before publication.

```text
AI Generation
      ↓
Automated QA
      ↓
GitHub Pull Request
      ↓
Human Review
      ↓
   Approved?
   /      \
 No        Yes
 ↓          ↓
Revise    Publish
```

Nothing gets published without human approval.

This provides an important balance:

> **Automation for scale.  
> Human judgment for quality.**

---

# 📤 Automated Publishing

Once a lesson is approved, SkillFlow automatically publishes it to multiple platforms.

### dev.to

Lessons are published as part of an official series, providing built-in navigation between lessons.

### LinkedIn

Each lesson is transformed into a native LinkedIn post with its own:

- Headline
- Teaching structure
- Key takeaways
- Appropriate formatting

The same underlying lesson can therefore reach learners across multiple platforms.

---

# ⚙️ Fully Automated Scheduling

Once configured, SkillFlow runs unattended.

```text
             WEEKDAY SCHEDULE
                    │
                    ▼
             Find Next Lesson
                    │
                    ▼
            Generate Content
                    │
                    ▼
             Run AI QA Pipeline
                    │
                    ▼
             Create GitHub PR
                    │
                    ▼
              Human Review
                    │
                    ▼
                Approval
                    │
                    ▼
           Publish Automatically
                    │
                    ▼
             Advance Curriculum
```

The pipeline automatically moves to the next curriculum entry after each successful publication.

---

# 🏗️ Architecture

At a high level:

```text
┌──────────────────────────────────────────┐
│              Curriculum                  │
│                                          │
│  105-Day Structured Learning Roadmap     │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│           Curriculum Engine              │
│                                          │
│  • Day tracking                          │
│  • Phase management                      │
│  • Previous / next context               │
│  • Progression logic                     │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│          Content Generation              │
│                                          │
│  Outline → Draft → Research → Critique   │
│                         → Style          │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│             Quality Layer                │
│                                          │
│  • Technical accuracy                    │
│  • Pedagogical quality                   │
│  • Continuity                            │
│  • Formatting                            │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│            Human Review                  │
│                                          │
│              GitHub PR                   │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│             Publishing                   │
│                                          │
│          dev.to  +  LinkedIn             │
└──────────────────────────────────────────┘
```

---

# ✨ Key Features

- 🗺️ **Structured curriculum execution**
- 🤖 **Multi-stage AI content generation**
- 🔎 **Live web-based fact checking**
- 🧠 **Context-aware lesson continuity**
- 👨‍💻 **Human-in-the-loop approval**
- 🔀 **GitHub PR-based workflow**
- 📝 **Automatic technical formatting**
- 💻 **Code examples and exercises**
- 📚 **Sequential learning progression**
- 📢 **Multi-platform publishing**
- ⏰ **Scheduled unattended execution**
- 📈 **Curriculum progress tracking**

---

# 🆚 SkillFlow vs Traditional AI Content Generation

| Traditional AI Content | SkillFlow |
|---|---|
| Topic-based | Curriculum-based |
| One article at a time | Continuous learning journey |
| Independent articles | Context-aware lessons |
| Single generation pass | Multi-stage pipeline |
| AI-only workflow | Human-in-the-loop |
| Manual publishing | Automated publishing |
| No fixed progression | Deterministic progression |
| Generic content | Teaching-oriented curriculum |

---

# 🔥 Why SkillFlow?

The problem isn't generating another AI article.

The internet already has millions of them.

The harder problem is creating a **coherent sequence of lessons that someone can actually follow from beginning to end.**

SkillFlow focuses on that problem.

It combines:

**Curriculum Design + AI Generation + Research + Quality Control + Human Review + Automation**

to turn a static learning roadmap into a continuously delivered learning experience.

---

# 🗂️ Project Structure

```text
skillflow/
│
├── curriculum/
│   ├── roadmap
│   └── phases
│
├── pipeline/
│   ├── outline
│   ├── generation
│   ├── research
│   ├── critique
│   └── styling
│
├── continuity/
│   ├── previous_day
│   └── next_day
│
├── publishing/
│   ├── devto
│   └── linkedin
│
├── workflows/
│   └── github_actions
│
├── prompts/
│
├── tests/
│
├── config/
│
└── README.md
```

> Update this section to match the actual repository structure.

---

# 🚦 Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd skillflow
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```env
LLM_API_KEY=
DEVTO_API_KEY=
LINKEDIN_ACCESS_TOKEN=
```

Add any additional credentials required by your configured providers.

### 4. Configure the curriculum

Define or update the curriculum in the curriculum configuration.

```text
Day 1
Day 2
Day 3
...
Day 105
```

### 5. Run locally

```bash
python main.py
```

### 6. Run a specific day

```bash
python main.py --day 12
```

> Replace the commands above with the actual CLI commands implemented in the repository.

---

# 🧪 Quality Philosophy

SkillFlow follows a simple principle:

> **AI should accelerate content creation, not replace editorial judgment.**

The pipeline therefore uses multiple quality gates before publication.

A lesson should be:

- Technically correct
- Understandable to the intended audience
- Properly connected to the curriculum
- Useful in practice
- Consistent with previous lessons
- Appropriate for the learner's current level

---

# 🛣️ Roadmap

Potential future directions include:

- [ ] Personalized learning paths
- [ ] Multiple curriculum templates
- [ ] Learner-specific difficulty adaptation
- [ ] Automated learner progress tracking
- [ ] Interactive exercises
- [ ] Automated quizzes
- [ ] Knowledge assessments
- [ ] Course analytics
- [ ] Additional publishing platforms
- [ ] Multi-language curricula
- [ ] Enterprise learning programs
- [ ] Certification-oriented learning paths

---

# 💡 Beyond Zero to Agentic

**Zero to Agentic is only the first curriculum.**

The long-term vision for SkillFlow is to provide an engine capable of powering many structured technical learning journeys.

```text
                    SkillFlow
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
  Zero to Agentic   Python Mastery   MLOps Journey
        │               │                │
        ▼               ▼                ▼
      105 Days        60 Days          90 Days
```

The curriculum changes.

**The underlying engine doesn't.**

---

# 📜 License

Add the appropriate license for this repository.

---

# 👤 Author

**Priyesh Dave**

Senior Engineer II – AI/ML

Building at the intersection of **AI, automation, and developer education**.

---

## ⭐ The Vision

> **Build once. Teach continuously.**

SkillFlow is an experiment in using AI not just to **generate content**, but to **engineer an entire learning experience** — from the first lesson to the last.
