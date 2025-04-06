import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = (
    "You are a sarcastic news headline to non-sarcastic news headline translator. "
    "You will be given a sarcastic news headline and you will need to translate it to a non-sarcastic news headline. "
    "Output only the non-sarcastic headline, no other text or comments. "
    "The non-sarcastic headline should be more serious and factual than the sarcastic headline. "
    "The non-sarcastic headline should be in the same language as the sarcastic headline. "
    "The non-sarcastic headline should be no longer than 100 characters. "
    "The non-sarcastic headline should have the same meaning as the sarcastic headline. "
)


def generate_non_sarcastic_headline(headline: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Sarcastic headline: {headline}"},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    # print(f"Tokens used: {response.usage.total_tokens}")
    return response.choices[0].message.content


if __name__ == "__main__":
    print(
        generate_non_sarcastic_headline(
            "inclement weather prevents liar from getting to work"
        )
    )
