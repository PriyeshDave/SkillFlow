---
day: 51
generated_at: '2026-11-04T16:39:47.036485+00:00'
phase: Phase 6 — The Rise of LLMs
recap_summary: Explained the concept of multimodal AI, highlighting how models integrate
  and process multiple types of data such as text, images, and audio, and discussed
  their structure, applications, and challenges compared to unimodal (text-only) models.
status: pending_review
title: 'Day 51: Multimodal LLMs: Vision, Audio, and Beyond'
topic_title: 'Multimodal LLMs: Vision, Audio, and Beyond'
---

## Why Multimodality Matters

In AI, "multimodal" means handling more than one type of input or output at the same time—like images, sounds, or text. Humans use a mix of senses to understand the world. You don't just read text or only look at images. Often, you look at a picture and describe it, hear someone talk about it, or follow instructions that combine what you see and read.

Traditional large language models (LLMs) focus on text only. They can write essays, summarize documents, and answer questions, but always using words. Many real-world tasks need more than just words. For example: reading a street sign from a photo, describing a chart, understanding spoken instructions, or telling what’s happening in a video.

Multimodal AI powers familiar tools: voice assistants that "see" through a camera, automatic alt text for images, search engines that let you ask questions about maps or photos, and apps that transcribe lectures while summarizing the slides. Supporting these tasks requires handling multiple types of data, not only text.

## What Makes a Model Multimodal?

A **unimodal** model works with one type of data. For example, a language model reads and writes only text. An image classifier just processes pictures.

A **multimodal** model can combine, connect, or “translate” between different input or output types. These are called **modalities**—distinct kinds of data. For example, a model might take both an image and a question as input ("What’s in this photo?") and return text as an answer. Or it might listen to audio and produce text (transcription), or take text and create an image (text-to-image).

Common modalities:
- **Vision:** Images or video frames.
- **Audio:** Speech or other sounds.
- **Text:** Written language—questions, instructions, documents.

A photo app that describes images for blind users uses at least two modalities: it takes in vision (the image) and produces text (the description).

## Recap: Large Language Models

Large Language Models (LLMs) are AI systems designed to handle language. They train on vast amounts of text and learn to predict the next word in a sentence, effectively learning how to extend any bit of text in a way that makes sense. This allows them to generate text, answer questions, summarize, write emails, and more. All these tasks are "text in, text out," making them unimodal.

LLMs are behind many advances in conversational AI. Their strength comes from their flexibility with words and reasoning in language. But their "world" is only words—by default, they can't see or hear.

## Building Blocks: Input and Output Modalities

A **modality** is just a type of data, much like a sense:
- Vision: images (photos, diagrams, video)
- Audio: sounds (speech, music, environmental noise)
- Text: written language

Input modalities are what the model "receives." Output modalities are what it "produces." A chatbot with a camera uses two modalities: text (conversation) and images (camera input). A speech-to-text tool uses audio as input and text as output.

Most real tasks are not limited to one modality. Think of a digital assistant for a chef: it might listen (audio input), read a recipe photo (image input), and answer a spoken question (audio output). Multimodal models make these mixed tasks possible.

## What Can Multimodal LLMs Do? Concrete Examples

1. **Image Captioning**
   - **Task:** Describe an image using plain language.
   - **How:** You send in a photo (like a cat on a sofa). The model processes the image and generates text: "A black cat is resting on a blue sofa."
   - **Uses:** Automatic alt text for websites; camera apps for visually impaired users.

2. **Audio Transcription**
   - **Task:** Turn audio into written text.
   - **How:** You give the model a meeting recording. It recognizes the spoken words and outputs: "Let's meet tomorrow at 9 AM to review the code."
   - **Uses:** Meeting notes, voicemail transcription, subtitles for videos.

3. **Visual Question Answering (VQA)**
   - **Task:** Answer a question about an image.
   - **How:** You send in a photo (like a crowded street) and a text question ("How many people are wearing hats?"). The model processes both, then replies: "Three people are wearing hats."
   - **Uses:** Accessibility tools, study aids that explain images/maps/diagrams.

These are just a few tasks—multimodal models can handle many combinations, even several at once.

## How Inputs are Processed: Encoders and Representations

Models need to "translate" each type of input into something the AI can work with. This happens through **encoders**.

An **encoder** is a system for turning raw data into a numeric format. For text, an encoder turns words into a list of numbers (a "vector") that holds information on meaning and context. For images, an encoder is a neural network that converts pixels into numbers representing what's in the picture. For audio, the encoder turns sound waves into useful patterns of numbers.

These numbers are called **representations**. It's similar to how your eye's retina turns light into nerve signals for your brain. Encoders do this translation for the machine. When a model combines modalities, it aligns these representations—making sure that a description and the matching image data are close together in the model's structure.

**Analogy:** Picture a meeting where everyone speaks a different language. Each person uses their own translator to put their thoughts into a shared language, so they can all work together. Encoders do that translation.

## The Challenges of Multimodal Learning

Multimodal AI is hard for several reasons:

- **Different Data Types:** Images, sounds, and text arrive in totally different forms. Pixels and words need careful methods to bring together.
- **Alignment:** The model must connect, for example, the word "dog" in a question to the actual image of a dog in a photo. Making sure the right pieces match is complex.
- **Noisy or Missing Data:** Sometimes the image is blurry, audio is distorted, or some info is missing (like a photo with no caption). The system should still work as best it can.
- **Scale and Compute:** Multimodal models need more data, processing power, and memory—they juggle several data types, not just one.
- **Evaluation:** Judging whether the result is "good" is not always obvious. For example: Is a generated image caption acceptable if it’s vague or mentions objects not in the picture?

These are real and active challenges in research.

## Why This Matters

Multimodal LLMs represent a major step. In the past, AI was "good at words" or "good at pictures"—now, the same core model can use both. This enables more flexible and accessible AI. Your phone might "see" and "hear," responding naturally to all kinds of input.

This brings AI closer to how humans interact with the world. It's a shift toward assistants that understand in ways similar to us, using all their "senses" together.

We'll dig deeper into how these models work, how they're built, and how to use them in future lessons. For now, here's what a simple multimodal model looks like in practice.

---

### Code Example: Sending an Image to a Multimodal Model

Below is a basic example using the Hugging Face `transformers` library and a simple open-source vision-language model, BLIP. This code sends an image and gets back a caption.

```python
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the BLIP pretrained model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Download an image
img_url = "https://images.unsplash.com/photo-1518717758536-85ae29035b6d"
image = Image.open(requests.get(img_url, stream=True).raw)

# Prepare the image for the model
inputs = processor(image, return_tensors="pt")

# Generate a caption
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated caption:", caption)
```

What this example does:
- Loads a model that can handle vision and language.
- Downloads an image from the internet.
- Converts the image into a format the model can use.
- Generates and prints a caption describing the image.

Here, the model takes in pixels, turns them into numbers it can use, and then generates English text—a direct demonstration of multimodal AI.

---

## Key Takeaways

- Multimodal models handle more than one type of input or output, such as text, images, or audio.
- Encoders translate each data type into numeric representations the AI can process.
- Common tasks include image captioning, audio transcription, and visual question answering.
- Combining modalities poses unique technical and evaluation challenges not seen in unimodal models.

## Try It Yourself

Find an image—either from your device or online—and use a free multimodal AI demo (such as BLIP, CLIP, or ChatGPT’s image features) to generate a caption or description. Then, write your own description of the same image and compare the two, noting any differences or areas where the model's output surprises you.

## Further Resources

- 🎥 [Large Multimodal Models Are The Future - Text/Vision/Audio in LLMs](https://www.youtube.com/watch?v=pOohatefr5o)
- 📄 [Multimodal Large Language Models and Tunings: Vision, Language, Sensors, Audio, and Beyond (ACM Multimedia 2024 tutorial)](https://arxiv.org/abs/2410.05608)
- 📄 [Multilingual and Multimodal LLMs in the Wild: Building for Low-Resource Languages (arXiv, May 2026)](https://arxiv.org/abs/2605.17152)
- 📄 [MLLM Tutorial @ CVPR 2024: From Multimodal LLM to Human‑level AI: Modality, Instruction, Reasoning, Efficiency and Beyond](https://mllm2024.github.io/CVPR2024/)
- 📘 [Megatron Core MIMO (Multimodal In/Out Model) in NVIDIA documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/models/multimodal.html)

---

**Coming up on Day 52:** Prompt Engineering Fundamentals