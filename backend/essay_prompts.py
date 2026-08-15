"""Prompt bank used only for building the synthetic-AI training data
(pipeline_02). Mixes Common-App-style personal statement prompts with
PERSUADE-style persuasive/argumentative prompts, since real admissions
essays span both registers and the DAIGT source data skews persuasive.
"""

PERSONAL_PROMPTS = [
    "Describe a challenge you overcame and what it taught you about yourself.",
    "Write about a time you failed at something and how you responded.",
    "Describe a person who has had a significant influence on you.",
    "Tell a story about a moment you realized something important about your identity.",
    "Describe an experience that changed how you see the world.",
    "Write about a risk you took and what happened as a result.",
    "Describe a topic, idea, or concept you find so engaging it makes you lose track of time.",
    "Reflect on a time you questioned or challenged a belief or idea.",
    "Describe a problem you'd like to solve, and why it's meaningful to you.",
    "Discuss an accomplishment or event that sparked personal growth.",
    "Write about a piece of art, music, or writing that changed you.",
    "Describe your background, identity, or interest that is so meaningful you feel your application would be incomplete without it.",
    "Tell us about a time you worked through a difficult situation with a friend or family member.",
    "Describe a place where you feel most yourself.",
    "Write about a tradition in your family or community that shaped who you are.",
    "Describe what community means to you and how you've contributed to one.",
    "Write about a time you had to advocate for yourself or someone else.",
    "Reflect on a lesson learned outside of the traditional classroom.",
    "Describe an unresolved question you continue to think about.",
    "Write about a moment when you felt most proud of yourself.",
]

PERSUASIVE_PROMPTS = [
    "Should schools require students to wear uniforms? Argue your position.",
    "Should students be allowed to use cell phones during the school day?",
    "Is it more beneficial for students to study a subject they excel in or one they struggle with?",
    "Should community service be a requirement for high school graduation?",
    "Should schools replace some in-person classes with online learning?",
    "Should the school day start later to accommodate teenage sleep schedules?",
    "Should extracurricular activities be mandatory for all students?",
    "Should students be able to grade their teachers?",
    "Is it better to work as a group or independently to solve problems?",
    "Should the voting age be lowered to sixteen?",
    "Should schools eliminate homework?",
    "Should students have a say in their school's curriculum?",
    "Should standardized testing be eliminated as a college admissions requirement?",
    "Should social media platforms be more strictly regulated for teenagers?",
    "Should schools teach financial literacy as a required course?",
]

ALL_PROMPTS = [(p, "personal") for p in PERSONAL_PROMPTS] + [(p, "persuasive") for p in PERSUASIVE_PROMPTS]

GENERATION_STYLES = {
    "direct": (
        "Write a college admissions essay responding to this prompt. "
        "Write in first person, 400 to 550 words, with a clear narrative and reflection.\n\nPrompt: {prompt}"
    ),
    "coached": (
        "You are a high school senior writing a compelling, specific personal statement for college "
        "applications. Respond thoughtfully to the prompt below, grounding your response in a believable "
        "personal anecdote with concrete detail. Aim for about 450 words.\n\nPrompt: {prompt}"
    ),
}

POLISH_STYLES = {
    "light_polish": (
        "Lightly copy-edit the following paragraph from a student's college admissions essay: fix grammar, "
        "improve word choice, and smooth the flow. Keep the same ideas, structure, and voice as much as "
        "possible. Return only the revised paragraph, nothing else.\n\nParagraph: {chunk}"
    ),
    "heavy_rewrite": (
        "Rewrite the following paragraph from a student's college admissions essay to sound more polished, "
        "sophisticated, and articulate, as a professional editor might. You may restructure sentences and "
        "elevate vocabulary. Keep the same core meaning. Return only the rewritten paragraph, nothing else.\n\n"
        "Paragraph: {chunk}"
    ),
}
