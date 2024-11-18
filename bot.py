from openai import OpenAI
from constants import APIKEY

# NVIDIA API Setup
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key= APIKEY
)

def generate_blog_post(topic):
    # Create a prompt for the chatbot
    prompt = f"""
Write a professional and detailed blog post about "{topic}". 

- Start with an engaging introduction that highlights the relevance of the topic and includes relatable examples. 
- Use a professional tone that is both informative and optimistic.
- Organize the blog with clear headings and subheadings, dividing the content into logical sections such as an introduction, technical details, use cases, benefits, and a conclusion. 
- Incorporate technical jargon where appropriate, and explain any complex terms or acronyms for clarity.
- Include examples or scenarios that help readers connect with the content.
- Conclude with a strong closing statement or call-to-action, summarizing the blog’s key points.
"""

    
    # Generate content using NVIDIA's API
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=1,
        max_tokens=1500,
        stream=True
    )
    
    # Collect the response
    blog_post = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            blog_post += chunk.choices[0].delta.content
    
    return blog_post

def main():
    # Accept user input
    topic = input("Enter the blog topic: ")
    
    print("\nGenerating content...\n")
    blog_post = generate_blog_post(topic)
    print(blog_post)

if __name__ == "__main__":
    main()
