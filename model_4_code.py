import openai
import numpy as np
from collections import Counter

def generate_data(task_type, num_samples, max_length, **kwargs):
    data = []
    if task_type == "addition":
        for _ in range(num_samples):
            num1 = np.random.randint(1, 10**max_length)
            num2 = np.random.randint(1, 10**max_length)
            data.append((f"{num1} + {num2}", str(num1 + num2)))
    elif task_type == "subtraction":
         for _ in range(num_samples):
              num1 = np.random.randint(1, 10**max_length)
              num2 = np.random.randint(1, 10**max_length)
              data.append((f"{max(num1,num2)} - {min(num1,num2)}", str(max(num1,num2) - min(num1,num2))))
    elif task_type == "multiplication":
         for _ in range(num_samples):
            num1 = np.random.randint(1, 10**2)
            num2 = np.random.randint(1, 10**2)
            data.append((f"{num1} * {num2}", str(num1 * num2)))
    elif task_type == "parity":
        for _ in range(num_samples):
            num = np.random.randint(1, 10**max_length)
            data.append((str(num), "even" if num % 2 == 0 else "odd"))
    elif task_type == "gsm8k": # Simplified
        for _ in range(num_samples):
            num1 = np.random.randint(1, 100)
            num2 = np.random.randint(1, 100)
            data.append((f"A fruit costs {num1} dollars, and another fruit costs {num2} dollars, how much would both cost?", str(num1+num2)))
    elif task_type == "gsm8k_hard": # Simplified
       for _ in range(num_samples):
           num1 = np.random.randint(1000, 100000)
           num2 = np.random.randint(1000, 100000)
           data.append((f"A fruit costs {num1} dollars, and another fruit costs {num2} dollars, how much would both cost?", str(num1+num2)))
    return data

def create_few_shot_prompt(task_type, few_shot_examples):
    prompt = ""
    for example in few_shot_examples:
        prompt += f"Q:{example[0]} A:{example[1]}\n"
    return prompt

def create_chain_of_thought_prompt(task_type, few_shot_examples):
    prompt = ""
    for example in few_shot_examples:
         prompt += f"Q: {example[0]}\n A: Let’s think step by step.\n{example[1]}\n"
    return prompt

def create_scratchpad_prompt(task_type, few_shot_examples):
    prompt = ""
    if task_type == "addition":
        for example in few_shot_examples:
            num1, num2 = example[0].split(" + ")
            result = int(num1) + int(num2)
            prompt += f"Q: {example[0]}\nA: First, {num1} + {num2} = {result}\nTherefore, the answer is {result}\n"
    return prompt

def create_algorithmic_prompt(task_type, few_shot_examples, details = "full"):
    prompt = ""
    if task_type == "addition":
        for example in few_shot_examples:
           num1, num2 = example[0].split(" + ")
           result = int(num1) + int(num2)
           prompt += f"Q:{example[0]}\nA: To calculate this, we first add the ones column, then the tens, then the hundreds etc until we have the final answer. If the ones digits are {num1} and {num2}, and we add these numbers we get {result}\nSo the answer is {result}\n"
    elif task_type == "subtraction":
        for example in few_shot_examples:
            num1, num2 = example[0].split(" - ")
            result = int(num1) - int(num2)
            prompt += f"Q: {example[0]}\nA: To calculate this, we subtract the ones column, then the tens, then the hundreds etc until we have the final answer. If the minuend is {num1} and the subtrahend is {num2}, and we perform the subtraction we get {result}\nTherefore the answer is {result}\n"
    elif task_type == "multiplication":
        for example in few_shot_examples:
            num1, num2 = example[0].split(" * ")
            result = int(num1) * int(num2)
            prompt += f"Q: {example[0]}\nA: To calculate this, we can use the standard multiplication algorithm, by considering the product of each pair of digits, then performing additions. So the answer is {result}\n"
    elif task_type == "parity":
         for example in few_shot_examples:
            num = example[0]
            result = "even" if int(num) % 2 == 0 else "odd"
            prompt += f"Q: {example[0]}\nA: To calculate this we take the modulo 2 of the number. The result is {result}\n"
    elif task_type == "gsm8k":
        for example in few_shot_examples:
           num1, num2 = example[0].split("costs ")[1].split(" dollars")[0], example[0].split("costs ")[2].split(" dollars")[0]
           result = int(num1) + int(num2)
           prompt += f"Q: {example[0]}\nA: To calculate the total cost we simply add the cost of the two fruits. The result is {result}\n"
    elif task_type == "gsm8k_hard":
        for example in few_shot_examples:
           num1, num2 = example[0].split("costs ")[1].split(" dollars")[0], example[0].split("costs ")[2].split(" dollars")[0]
           result = int(num1) + int(num2)
           prompt += f"Q: {example[0]}\nA: To calculate the total cost we simply add the cost of the two fruits. The result is {result}\n"
    return prompt

def call_openai_api(prompt, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=prompt,
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].text.strip()

def evaluate_output(output, expected_output, metrics, task_type):
    results = {}
    if "Accuracy" in metrics:
        results["Accuracy"] = 1 if output == expected_output else 0

    return results

def evaluate_model(data, prompt, api_key, task_type, prompt_type):
     scores = []
     for input_str, expected_output in data:
        full_prompt = prompt + f"Q: {input_str}\n A:"
        llm_output = call_openai_api(full_prompt, api_key)
        if task_type == "parity":
           output_str = llm_output.split("So the answer is ")[-1]
        else:
           output_str = llm_output.split("The result is ")[-1].split("The answer is")[-1].strip()
        if prompt_type == "Chain-of-Thought":
            output_str = llm_output.split("step by step.")[-1].split("\n")[-1].split("So the answer is")[-1].split("The result is")[-1].strip()
        scores.append(evaluate_output(output_str, expected_output, ["Accuracy"], task_type))
     return scores

if __name__ == '__main__':
    api_key = "YOUR_API_KEY"
    tasks = ["addition", "subtraction", "multiplication", "parity", "gsm8k", "gsm8k_hard"]
    num_samples = 10
    max_length = 2

    for task in tasks:
      train_data = generate_data(task, 5, max_length)
      test_data = generate_data(task, 5, max_length)

      prompt_algorithmic = create_algorithmic_prompt(task, train_data)
      prompt_cot = create_chain_of_thought_prompt(task, train_data)
      prompt_few_shot = create_few_shot_prompt(task, train_data)
      
      algorithmic_scores = evaluate_model(test_data, prompt_algorithmic, api_key, task, "algorithmic")
      cot_scores = evaluate_model(test_data, prompt_cot, api_key, task, "Chain-of-Thought")
      few_shot_scores = evaluate_model(test_data, prompt_few_shot, api_key, task, "Few Shot")

      print(f"Algorithmic Scores on task {task}: ", [d["Accuracy"] for d in algorithmic_scores])
      print(f"Chain of Thought Scores on task {task}: ", [d["Accuracy"] for d in cot_scores])
      print(f"Few Shot Scores on task {task}: ", [d["Accuracy"] for d in few_shot_scores])
