'''



'''

import re

def prompt(context, passage, tokenizer):
    dialog = [
        {
            'role':'user',
            'content': f"""\
Please read the following text, and simulate a dialog between a Student and Teacher. The Student asks intelligent questions, covering all the important knowledge from the passage. The Teacher gives long and thoughtful answers. They should not talk about the passage directly. They should have a dialog where the content of their communication is grounded in the passage. Let's warm up with a short passage:

Context:

{context}

Passage:

Stoicism became particularly fashionable in the Roman period. Although never identifying as a Stoic himself, Cicero, who studied philosophy in Athens and endeavored to popularize it at Rome, engaged extensively with Stoic theory in his philosophical works and modeled his On Proper Functions (De Officiis) on Panaetius’ treatise of the same name. During the Imperial era, several prominent public figures were associated with the Stoic school. Appointed to the court of Emperor Augustus in the late 1st century BCE was the Stoic philosopher Arius Didymus, and a generation later Seneca served as advisor to Nero."""
        },
        {
            'role':'assistant',
            'content': """\
Understood. I should simulate a dialog between a Teacher and Student, and the dialog should be supported by the provided passage, but not mention the actual passage itself. Here is a hypothetical conversation:

[Student] Please tell me about On Proper Functions (De Officiis).

[Teacher] This was a philosophical work written by Cicero during the Roman period. At this time, Stoicism was becoming a fashionable movement, and while Cicero never identified as a stoic himself, he engaged with the theory extensively in his work On Proper Functions (De Officiis). This work in turn was modeled on Panaetius' treatise of the same name.

[Student] Who were some of the actual Stoics around this time?

[Teacher] Arius Didymus was associated with the Stoic school. He served in the court of Emperor Augustus during the Imperial era in the late 1st century BCE. A generation later, Nero's advisor Seneca was another prominent Stoic figure."""
        },
        {
            'role':'user',
            'content': f"""
Nice work. It sticks to the given details well which is critical. We need to ground the discussion in facts from the passage. For our next passage, allow the Student to ask more open ended questions, and allow the Teacher to give longer and more detailed answers. If there is enough grounding information, it is proper for the Teacher to give a lecture on all the material, and quote passages, though of course never mentioning the passage directly. Here is the next passage for you to convert to dialog:

Context:

{context}

Passage:

{passage}"""
        },
        {
            'role':'assistant',
            'content': f"""
I see. I will focus on having the Student ask more broad questions, and the Teacher will answer open ended questions at great length, and in great detail like part of a lecture. Above all we should ground the discussion on the original passage and not infer too much. We should also not reference the passage directly, but instead present the information as if the Teacher is delivering a lesson, and occassionally quoting material. Now let's generate a dialog for your new passage:

[Student] """
        },
    ]
    p = tokenizer.apply_chat_template(
        dialog, tokenize=False, return_tensors=False, add_generation_prompt=True
    )

    # Remove end token from the chat template, to allow assistant to keep
    # writing, itself.
    p = p[:-len(' </s>')]
    return p



def extract_dialog_pairs(dialog):
    """
    Extracts pairs of questions and responses from a dialog and returns them as a list of dictionaries.
    The keys in the dictionary are 'role' (either 'user' or 'assistant') and 'content' (text of the dialog).

    :param dialog: A string containing the dialog.
    :return: A list of dictionaries with the dialog pairs.
    """
    # Regular expression pattern to match the dialog pairs
    pattern = r"\[(student|teacher)\](.*?)(?=\[student\]|\[teacher\]|\<\/s\>|$)"

    # Finding all matches
    matches = re.findall(pattern, dialog, re.DOTALL | re.IGNORECASE)

    dialog_pairs = []
    for role, content in matches:
        if role.lower() == 'teacher':
            role = 'assistant'
        elif role.lower() == 'student':
            role = 'user'
        dialog_pairs.append({'role': role, 'content': content.strip()})

    return dialog_pairs
