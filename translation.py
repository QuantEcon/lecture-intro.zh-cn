"""
Before running the file, config the environment by adding the OpenAI API key:


echo "export OPENAI_API_KEY='yourkey'" >> ~/.zshrc
source ~/.zshrc
echo $OPENAI_API_KEY # Test to see if it is added
"""


import openai
import os
from concurrent.futures import ThreadPoolExecutor

def process_file(filename, function, assistant_cn_id):
    input_file = os.path.join(directory, filename)
    print(f'processing {input_file}')
    if os.path.isfile(input_file):
        function(input_file, assistant_cn_id)


def split_text(content, chunk_size=3000):
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size

        # If we are at the end of the content, just append the rest
        if end >= len(content):
            chunks.append(content[start:])
            break

        # Find the nearest line break before the chunk size
        next_line_break = content.rfind('\n', start, end)
        if next_line_break == -1:
            # If no line break is found within the chunk size, extend to the end
            next_line_break = end

        # Check if a code cell starts within the chunk
        code_cell_start = content.find('```{code-cell}', start, next_line_break)
        if code_cell_start != -1:
            # If a code cell starts, find its end
            code_cell_end = content.find('```', code_cell_start + 14)
            if code_cell_end != -1:
                # Move the end to the end of the code cell
                next_line_break = content.find('\n', code_cell_end) + 1

        chunks.append(content[start:next_line_break].strip())
        start = next_line_break

    return chunks

def translate_cn(input_file, assistant_id):
    # Initialize the OpenAI client
    client = openai.OpenAI()

    # Read the content of the input markdown file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into chunks
    chunks = split_text(content, chunk_size=3000)

    # Initialize the OpenAI client and thread
    thread = client.beta.threads.create()

    translated_content = ""

    for chunk in chunks:
        # Create and poll the run for each chunk
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="Please translate the following content into simplified Chinese. Maintain all the markdown syntax and directives unchanged. Only translate text and code comments Give the results directly without system messages: " + chunk
        )
        
        if run.status == 'completed': 
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            translated_content += '\n\n' + messages.data[0].content[0].text.value
            print(translated_content)
        else:
            print(f"Translation failed for chunk: {chunk[:50]}... Status: {run.status}")
            continue

    # Create the output file name
    output_file = input_file.replace('.md', '_cn.md')

    # Write the translated content to the new markdown file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(translated_content)

    print(f"Translated content has been saved to {output_file}")


if __name__ == "__main__":
    directory = "lectures"
    assistant_cn_id = 'asst_zjzyGwEZ1rVuJYWNQk6nzQTA'

    files = [f for f in os.listdir(directory) if f.endswith('.md') and os.path.isfile(os.path.join(directory, f))]
    print(openai.beta.assistants.list())
    
    print(f'files to translate: {files}')
    
    file_handler = lambda file: process_file(file, translate_cn, assistant_cn_id)
    with ThreadPoolExecutor() as executor:
        executor.map(file_handler, files[1:])