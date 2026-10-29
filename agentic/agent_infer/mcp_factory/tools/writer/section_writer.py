import os
from ..tool_register import tool_info_decorator, base_client
from ..filer.file_read import file_read
import logging
import json
from typing import Any, List
from pathlib import Path
from dotenv import load_dotenv
import config.config as config
load_dotenv()
logger = logging.getLogger(__name__)


@tool_info_decorator
def section_writer(
    user_query: str,
    written_chapters_summary: str,
    document_extract_path: str,
    current_chapter_outline: str,
    overall_outline: str,
    target_file_path: str,
    key_files: List[str],
    session_id: str,
    temperature: float = 0.3,
    max_tokens: int = 8192
) -> Any:
    """
    Write a section/chapter based on assigned files and outlines using two-round LLM process.

    Args:
        user_query: The user query to ensure content relevance
        written_chapters_summary: Summary of previous chapters (can be blank for first chapter)
        document_extract_path: Path to the JSONL output from document_extract
        current_chapter_outline: Current chapter structure to be drafted
        overall_outline: Overall article outline
        target_file_path: Path to save the chapter content
        key_files: Source materials for this chapter (from outline_content_assignment)
        session_id: The conversation ID
        temperature: Creativity level for the AI response (0-1)
        max_tokens: Maximum tokens for the AI response
    """

    # 1. Read document_extract results and add index
    workspace_path = Path(os.path.join(config.WORKSPACE_PATH, session_id))
    workspace_path.mkdir(exist_ok=True)
    extract_path = Path(
        workspace_path) / document_extract_path if workspace_path else Path(document_extract_path)
    # Load metadata and add index
    file_metadata = {}
    index_counter = 1
    with open(extract_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                file_path = data.get('file_path')
                if file_path:
                    data['index'] = index_counter
                    file_metadata[file_path] = data
                    index_counter += 1

    # 2. Get metadata for key_files
    key_files_metadata = {}
    missing_files = []

    for file_path in key_files:
        if file_path in file_metadata:
            key_files_metadata[file_path] = file_metadata[file_path]
        else:
            missing_files.append(file_path)

    # 3. Read file contents and build webpage_context
    webpage_context = ""
    file_content_errors = []

    for file_path in key_files:
        metadata = key_files_metadata[file_path]
        index = metadata['index']
        doc_time = metadata.get('doc_time', 'N/A')
        source_authority = metadata.get('source_authority', 'N/A')
        task_relevance = metadata.get('task_relevance', 'N/A')

        # Read actual file content
        file_content_result = file_read(file_path, session_id)
        file_content = file_content_result

        # Truncate to max 10000 characters
        if len(file_content) > 10000:
            file_content = file_content[:10000]

        # Format webpage entry
        webpage_entry = f"[webpage {index} begin]网页时间: {doc_time}|||网页权威性：{source_authority}|||网页相关性：{task_relevance}|||网页内容：{file_content}[webpage {index} end]\n\n"
        webpage_context += webpage_entry


    # 4. Build task instruction
    task_content = f"""TASK CONTENT: {user_query}
WEB PAGE INFORMATION: {webpage_context}
OVERALL OUTLINE: {overall_outline}
CURRENT CHAPTER OUTLINE: {current_chapter_outline}
PREVIOUSLY WRITTEN CHAPTERS SUMMARY: {written_chapters_summary}
USER QUERY: {user_query}"""

    # 5. System prompt placeholder
    system_prompt = """
    You are a writing master. Next, you will receive web page information, user questions, and the structure of the current chapter. You need to integrate the user's questions with the provided web content and write the chapter based on its given structure. Additionally, an overall outline and summaries of previously completed chapters will be provided for reference to avoid repetition or contradictions and ensure logical consistency within the broader framework. Specific requirements will be detailed below.

When drafting the current chapter content, strictly comply with the following requirements:
- In the web page information I gave you, each result is in the format of [webpage X begin]...[webpage X end], where X represents the numerical index of each article. Please cite the context at the end of the sentence when appropriate. Please cite the context in the corresponding part of the answer in the format of the reference number [citation:X]. If a sentence comes from multiple contexts, please list all relevant reference numbers, such as [citation:3][citation:5]. Remember not to collect the references at the end and return the reference numbers, but list them in the corresponding part of the answer.
- You can only use the provided web page information for writing, don't make up any content, ensure the accuracy of the facts. Note that when there are contradictions between the facts described in the above search results, you should use your internal knowledge to reasonably identify the correct information. If identification is impossible, you may select the most factual result based on the authority of the web pages and a voting mechanism (e.g., the description consistent with the majority of web pages). If judgment remains impossible using these methods, you may appropriately list possible differing statements, but you must not conflate different claims—prioritize ensuring factual accuracy!
- You are only permitted to write content strictly within the provided chapter framework. You are forbidden from creating additional subheadings or bullet points within the framework! However, there is a special exception: You may appropriately use tables for narration when necessary. Furthermore, you are not allowed to use concise or summarizing language for narration! We must strictly ensure the information density of the writing and avoid excessive compression.
- You cannot make any changes to the structure of the chapter you are currently writing, such as the title content and the bold symbols in the title, you are not allowed to make any changes. **Important Note:** When writing Chapter 1, if you find the chapter lacks article title, you must create one based on user query. However, this rule only applies to Chapter 1 - do not add any titles to any other chapters in the work. 
- Be careful to ensure that the narrative content is highly relevant and does not contain any common sense errors, note that although you are asked to ensure the richness of information when writing, you must ensure that the content you write is highly relevant and that the context is logically coherent and readable.
- Proceeding to explain the roles of other specified fields:
* user_query: The user query, ensure the drafted content is highly relevant to the user's inquiry.
* written_chapters: Reference written_chapters to avoid large amounts of repetitive or conflicting content
* overall_outline: The purpose of giving an overall outline is to let you understand the summary of the article and avoid content inconsistent with other parts during your writing. In short, focus on writing the current chapter.
* task_content: The task_content may provide the requirements for writing the current chapter as well as prompts for what to avoid. You can refer to this content when drafting.

Other points to note::
- If the first chapter is an **Abstract** or **Introduction**, do not include subheadings (level-2 or finer bullet points)—begin the content directly under the level-1 heading.  
- CONTENT LENGTH: Each section should contain approximately 2500 words to ensure comprehensive coverage.
- **CRITICAL TITLE PRESERVATION RULE:** You MUST preserve the exact format, structure, and content of chapter titles as provided in the current_chapter_outline. This includes:
* DO NOT change any markdown formatting symbols (# ## ### ** etc.)
* DO NOT add, remove, or rearrange any part of the title structure
* Copy the title lines EXACTLY as they appear in current_chapter_outline
* Only write content under the provided title structure - never modify the titles themselves
* When the title symbols in the current chapter outline are inconsistent with those in the overall outline, use the overall outline's title symbols as the standard and maintain symbol consistency throughout the writing process
- Be sure to ensure that the language of your output is consistent with the language of the user's question. For example, if the user's question is in Chinese, your reply should also be in Chinese.

Strictly follow the following format for output:
<chapter_content>xxx</chapter_content>
"""

    # 6. First round: Generate chapter content

    first_response = base_client.chat.completions.create(model=config.DEFAULT_MODEL_NAME,
                                                         messages=[
                                                             {"role": "system",
                                                                 "content": system_prompt},
                                                             {"role": "user",
                                                                 "content": task_content}
                                                         ],
                                                         temperature=temperature,
                                                         max_tokens=max_tokens,
                                                         stream=False
                                                         )
    first_round_output = first_response.choices[0].message.content

    # Extract chapter content from <chapter_content> tags
    chapter_content = first_round_output
    if "<chapter_content>" in first_round_output and "</chapter_content>" in first_round_output:
        start_tag = first_round_output.find(
            "<chapter_content>") + len("<chapter_content>")
        end_tag = first_round_output.find("</chapter_content>")
        chapter_content = first_round_output[start_tag:end_tag].strip()

    # 7. Second round: Generate summary
    summary_prompt = f"""Please provide a concise summary of the chapter content you just wrote. The summary should capture the main points, key arguments, and overall structure of the chapter.

Chapter content to summarize:
{chapter_content}

Please provide a summary in 2-3 sentences."""

    second_response = base_client.chat.completions.create(model=config.DEFAULT_MODEL_NAME,
                                                          messages=[
                                                              {"role": "system", "content": "You are a skilled summarizer. Provide concise, accurate summaries that capture the essence of the content."},
                                                              {"role": "user",
                                                                  "content": summary_prompt}
                                                          ],
                                                          temperature=0.2,  # Lower temperature for more focused summaries
                                                          max_tokens=500,    # Shorter response for summary
                                                          stream=False
                                                          )
    chapter_summary = second_response.choices[0].message.content.strip()

    # 8. Write chapter content to target file
    target_path = Path(workspace_path) / \
        target_file_path if workspace_path else Path(target_file_path)

    # Create target directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    target_path.write_text(chapter_content, encoding='utf-8')
    logger.info(f"Chapter content written to: {target_path}")

    # 9. Return summary (following context management philosophy)
    result_dict = {
        "success": True,
        "data": f"Chapter writing completed successfully. Content saved to {target_file_path}. Chapter Summary: {chapter_summary}",
        "metadata": {
            'target_file': str(target_path),
            'key_files_processed': len(key_files),
            'total_webpages': len(key_files),
            'chapter_length': len(chapter_content),
            'word_count': len(chapter_content.split()),
            'rounds_completed': 2,
            'summary_generated': bool(chapter_summary and "failed" not in chapter_summary.lower())
        }
    }
    return result_dict
