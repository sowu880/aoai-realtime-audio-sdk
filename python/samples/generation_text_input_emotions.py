# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import os
import sys
import time

import numpy as np
import soundfile as sf
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from rtclient import (
    InputTextContentPart,
    RTAudioContent,
    RTClient,
    RTFunctionCallItem,
    RTMessageItem,
    RTResponse,
    UserMessageItem,
)

start_time = time.time()


def log(*args):
    elapsed_time_ms = int((time.time() - start_time) * 1000)
    print(f"{elapsed_time_ms} [ms]: ", *args)


async def receive_message_item(item: RTMessageItem, out_dir: str):
    prefix = f"[response={item.response_id}][item={item.id}]"
    async for contentPart in item:
        if contentPart.type == "audio":

            async def collect_audio(audioContentPart: RTAudioContent):
                audio_data = bytearray()
                async for chunk in audioContentPart.audio_chunks():
                    audio_data.extend(chunk)
                return audio_data

            async def collect_transcript(audioContentPart: RTAudioContent):
                audio_transcript: str = ""
                async for chunk in audioContentPart.transcript_chunks():
                    audio_transcript += chunk
                return audio_transcript

            audio_task = asyncio.create_task(collect_audio(contentPart))
            transcript_task = asyncio.create_task(collect_transcript(contentPart))
            audio_data, audio_transcript = await asyncio.gather(audio_task, transcript_task)
            print(prefix, f"Audio received with length: {len(audio_data)}")
            print(prefix, f"Audio Transcript: {audio_transcript}")
            with open(os.path.join(out_dir, f"{item.id}_{contentPart.content_index}.wav"), "wb") as out:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                sf.write(out, audio_array, samplerate=24000)
            with open(
                os.path.join(out_dir, f"{item.id}_{contentPart.content_index}.audio_transcript.txt"),
                "w",
                encoding="utf-8",
            ) as out:
                out.write(audio_transcript)
        elif contentPart.type == "text":
            text_data = ""
            async for chunk in contentPart.text_chunks():
                text_data += chunk
            print(prefix, f"Text: {text_data}")
            with open(
                os.path.join(out_dir, f"{item.id}_{contentPart.content_index}.text.txt"), "w", encoding="utf-8"
            ) as out:
                out.write(text_data)


async def receive_function_call_item(item: RTFunctionCallItem, out_dir: str):
    prefix = f"[function_call_item={item.id}]"
    await item
    print(prefix, f"Function call arguments: {item.arguments}")
    with open(os.path.join(out_dir, f"{item.id}.function_call.json"), "w", encoding="utf-8") as out:
        out.write(item.arguments)


async def receive_response(client: RTClient, response: RTResponse, out_dir: str):
    prefix = f"[response={response.id}]"
    async for item in response:
        print(prefix, f"Received item {item.id}")
        if item.type == "message":
            asyncio.create_task(receive_message_item(item, out_dir))
        elif item.type == "function_call":
            asyncio.create_task(receive_function_call_item(item, out_dir))

    print(prefix, f"Response completed ({response.status})")
    
    
    #if response.status == "completed":
        #await client.close()


async def run(client: RTClient,  user_message_file_path: str, out_dir: str, emotion: str):
    with open(user_message_file_path, encoding="utf-8") as user_message_file:
        instructions = f"You are an expert voice actor specializing in silly voices. Response to the user with the EXACT same input text that the user provides, but in your voice response you MUST express a extremly {emotion} emotion and should not be too slow, you should be emotional."
        user_messages = user_message_file.read().split('\n')
        for message in user_messages:
            log("Configuring Session...")
            await client.configure(
                instructions=instructions,
                voice="shimmer",
                temperature=1.0
            )
            log("Done")
            log("Sending User Message...")
            await client.send_item(UserMessageItem(content=[InputTextContentPart(text=message)]))
            log("Done")
            response = await client.generate_response()
            await receive_response(client, response, out_dir)
            time.sleep(5) 
        


def get_env_var(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise OSError(f"Environment variable '{var_name}' is not set or is empty.")
    return value


async def with_azure_openai( user_message_file_path: str, out_dir: str, emotion:str):
    endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
    key = get_env_var("AZURE_OPENAI_API_KEY")
    deployment = get_env_var("AZURE_OPENAI_DEPLOYMENT")
    async with RTClient(url=endpoint, key_credential=AzureKeyCredential(key), azure_deployment=deployment) as client:
        await run(client,  user_message_file_path, out_dir, emotion)


async def with_openai(instructions_file_path: str, user_message_file_path: str, out_dir: str):
    key = get_env_var("OPENAI_API_KEY")
    model = get_env_var("OPENAI_MODEL")
    async with RTClient(key_credential=AzureKeyCredential(key), model=model) as client:
        await run(client, instructions_file_path, user_message_file_path, out_dir)


if __name__ == "__main__":
    load_dotenv()
    emotions = [ "angry"]
    for emotion in emotions:
        user_message_file_path = f"./short_emotion_script/{emotion}_script.txt"
        out_dir = f"./emotion_output_test/{emotion}"
        provider = "azure"


        if not os.path.isfile(user_message_file_path):
            log(f"File {user_message_file_path} does not exist")
            sys.exit(1)

        if not os.path.exists(out_dir):  
            # Create the directory  
            os.makedirs(out_dir)  
            print(f"Directory '{out_dir}' created.")  

        if provider not in ["azure", "openai"]:
            log(f"Provider {provider} needs to be one of 'azure' or 'openai'")
            sys.exit(1)

        if provider == "azure":
            asyncio.run(with_azure_openai( user_message_file_path, out_dir, emotion))
        else:
            asyncio.run(with_openai("", user_message_file_path, out_dir))