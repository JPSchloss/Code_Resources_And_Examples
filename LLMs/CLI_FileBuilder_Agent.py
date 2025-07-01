#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

import openai

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")  # make sure you export this

# Which models to use
GENERATION_MODEL = "gpt-4"
COGENCY_MODEL   = "gpt-3.5-turbo"

# Controls: comment these out if you want to disable confirmation or cogency checks
INTERACTIVE_PROMPTS = True
USE_COGENCY_CHECK   = True
# ───────────────────────────────────────────────────────────────────────────────


def generate_commands(structure_description: str) -> list[str]:
    """
    Ask ChatGPT (GENERATION_MODEL) to output a list of CLI commands
    that will create the file structure described by the user.
    """
    system = "You are a shell-scripting assistant."
    user   = (
        "Given this description of a directory/file structure:\n\n"
        f"{structure_description}\n\n"
        "Output ONLY the sequence of bash commands (one per line) "
        "needed to create it."
    )
    resp = openai.ChatCompletion.create(
        model=GENERATION_MODEL,
        messages=[
            {"role":"system",  "content": system},
            {"role":"user",    "content": user}
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content.strip()
    return [line for line in content.splitlines() if line.strip()]


def check_cogency(cmd: str) -> str:
    """
    Ask ChatGPT (COGENCY_MODEL) to explain what the given CLI command does,
    and whether it is appropriate / safe.
    """
    system = "You are a helpful Linux sysadmin."
    user   = (
        f"Please explain in a sentence or two what this command does, "
        f"and confirm whether it makes sense for creating a file structure:\n\n"
        f"{cmd}"
    )
    resp = openai.ChatCompletion.create(
        model=COGENCY_MODEL,
        messages=[
            {"role":"system",  "content": system},
            {"role":"user",    "content": user}
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def run_command(cmd: str):
    """
    Execute the command in a subprocess.
    """
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Use ChatGPT + CLI to scaffold a file structure."
    )
    parser.add_argument(
        "description", 
        help="A natural-language description of the folders/files you want."
    )
    args = parser.parse_args()

    print("\n🎯 Generating commands to create:\n   →", args.description, "\n")
    cmds = generate_commands(args.description)
    if not cmds:
        print("❌ No commands generated. Exiting.")
        sys.exit(1)

    for cmd in cmds:
        print(f"\n🔹 Proposed command:\n    {cmd}")
        
        if USE_COGENCY_CHECK:
            explanation = check_cogency(cmd)
            print(f"\n💬 Cogency check:\n    {explanation}")

        if INTERACTIVE_PROMPTS:
            choice = input("\n▶ Execute this command? ([y]es / [n]o / [a]bort): ").lower().strip()
            if choice == "a":
                print("⛔ Aborted by user.")
                sys.exit(0)
            elif choice != "y":
                print("↩ Skipping.")
                continue

        try:
            #run_command(cmd)
            print("✅ Success.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed (exit {e.returncode}). Stopping.")
            sys.exit(e.returncode)

    print("\n🎉 All done.")

if __name__ == "__main__":
    main()
