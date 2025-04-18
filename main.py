from enum import Enum
from pathlib import Path

from pydantic import BaseModel, StringConstraints, conint
from typing_extensions import Annotated, Optional

from distilabel.models import LlamaCppLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration


# class Weapon(str, Enum):
#     sword = "sword"
#     axe = "axe"
#     mace = "mace"
#     spear = "spear"
#     bow = "bow"
#     crossbow = "crossbow"


# class Armor(str, Enum):
#     leather = "leather"
#     chainmail = "chainmail"
#     plate = "plate"
#     mithril = "mithril"


# class Character(BaseModel):
#     name: Annotated[str, StringConstraints(max_length=30)]
#     age: conint(gt=1, lt=3000)
#     armor: Armor
#     weapon: Weapon

class Contact(BaseModel):
    name: str
    email: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip: Optional[str]

# Download the model with
# curl -L -o ~/Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf

model_path = "Downloads/openhermes-2.5-mistral-7b.Q4_K_M.gguf"

with Pipeline("RPG-characters") as pipeline:
    system_prompt = (
        "You are a contact. You have seen thousands of different people and their attributes."
        " Please return a JSON object with common attributes of a contact from a CRM."
    )

    load_dataset = LoadDataFromDicts(
        name="load_instructions",
        data=[
            {
                "system_prompt": system_prompt,
                "instruction": "Give me a new random contact",
            }
            for _ in range(10)
            # {
            #     "system_prompt": system_prompt,
            #     "instruction": f"Give me a character description for a {char}",
            # }
            # for char in ["dwarf", "elf", "human", "ork"]
        ],
    )
    llm = LlamaCppLLM(
        model_path=str(Path.home() / model_path),  # type: ignore
        n_gpu_layers=0,  # Disable GPU acceleration
        n_ctx=2048,  # Use a smaller context window
        structured_output={"format": "json", "schema": Contact},
    )
    # llm = OllamaLLM(
    #     model="qwen2:7b",
    #     # model_path=str(Path.home() / model_path),  # type: ignore
    #     # n_gpu_layers=-1,
    #     # n_ctx=1024,
    #     structured_output={"format": "json", "schema": Character},
    # )
    # Change to vLLM as such:
    # llm = vLLM(
    #     model="teknium/OpenHermes-2.5-Mistral-7B",
    #     extra_kwargs={"tensor_parallel_size": 1},
    #     structured_output={"format": "json", "schema": Character},
    # )

    text_generation = TextGeneration(
        name="text_generation_rpg",
        llm=llm,
        input_batch_size=8,
        output_mappings={"model_name": "generation_model"},
    )
    load_dataset >> text_generation


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            text_generation.name: {
                "llm": {"generation_kwargs": {"max_new_tokens": 256}}
            }
        },
        use_cache=False,
    )
    for num, character in enumerate(distiset["default"]["train"]["generation"]):
        print(f"Character: {num}")
        print(character)
