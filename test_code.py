import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
import os
from dspy.primitives.prediction import Prediction
turbo = dspy.EB(model="ernie-3.5", max_output_tokens=300)
dspy.settings.configure(lm=turbo)
from dspy.predict import KNN
# Load math questions from the GSM8K dataset
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[115:135], gsm8k.dev[50:52]

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ProgramOfThought(GenerateAnswer)
    
    def forward(self, question):
        return self.prog(question=question)

class ZeroShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = turbo
    
    def forward(self, question):
        completions = self.prog(prompt=question)
        return Prediction.from_completions(completions, signature="question -> answer")


from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPRO, KNNFewShot, BootstrapFewShotWithOptuna
from dspy.teleprompt.ensemble import Ensemble
from dspy.teleprompt import BootstrapFewShotWithOptuna
# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=0, max_labeled_demos=0)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
# teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric, max_bootstrapped_demos=4, num_candidate_programs=1, num_threads=1)
# teleprompter = MIPRO(prompt_model=turbo, task_model=turbo, metric=gsm8k_metric, view_data_batch_size=2)
# teleprompter = BootstrapFewShotWithOptuna(metric=gsm8k_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=1)

kwargs = dict(num_threads=1, display_progress=True, display_table=0)
# optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, eval_kwargs=kwargs, max_bootstrapped_demos=2, max_labeled_demos=2, num_trials=1)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)

# knn_optimizer = KNNFewShot(KNN, k=3, trainset=gsm8k_trainset)

# s = CoT()
# optimized_cot = knn_optimizer.compile(student=s, trainset=gsm8k_trainset, valset=gsm8k_devset)
# optimized_cot(question="Amanda has to sell 80 tickets in 3 days so she can make enough money to go on vacation. On the first day, she sells 5 of her friends 4 tickets each. On the second day she sells 32 tickets. How many tickets does she need to sell on the third day to meet her goal?")
breakpoint()
from dspy.evaluate import Evaluate

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, display_progress=True, display_table=0, num_threads=1)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)
# print(turbo.inspect_history(n=1))
import time
s_time = time.time()
res = optimized_cot(question="In seven years, Talia will be 20 years old.  Talia's mom is currently three times as old as Talia is today.  In three years, Talia's father will be the same age as Talia's mom is today.  Currently, how many years old is Talia's father?")
print(f"Cost Time: {time.time() - s_time}s")
breakpoint()
