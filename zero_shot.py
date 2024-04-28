import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
import os
from dspy.primitives.prediction import Prediction
turbo = dspy.EB(model="ernie-3.5", max_output_tokens=200)
dspy.settings.configure(lm=turbo)

# Load math questions from the GSM8K dataset
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[110:120], gsm8k.dev[20:22]

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

class ZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = turbo
    
    def forward(self, question):
        completions = self.prog(prompt=question)
        return Prediction.from_completions(completions, signature="question -> answer")


from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPRO

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=0, max_labeled_demos=0)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
kwargs = dict(num_threads=1, display_progress=True, display_table=0)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, display_progress=True, display_table=0, num_threads=1)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)
# print(turbo.inspect_history(n=1))
import time
s_time = time.time()
res = optimized_cot("In seven years, Talia will be 20 years old.  Talia's mom is currently three times as old as Talia is today.  In three years, Talia's father will be the same age as Talia's mom is today.  Currently, how many years old is Talia's father?")
print(f"Cost Time: {time.time() - s_time}s")
breakpoint()