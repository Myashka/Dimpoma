import numpy as np
import torch
import wandb
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class Q_A_Dataset(Dataset):
    """
    Dataset class to create dataset from pandas DataFrame that has columns Q_Body, Q_title and A_Body.
    Returns dict with "input_ids", "attention_mask", "labels", "answers", "questions", "titles".

    params:
        df[pd.DataFrame] - DataFrame from need to create dataset

        tokenizer

        promt_1

        promt_2

        promt_3
    """

    def __init__(
        self,
        df,
        tokenizer,
        promt_1,
        promt_2,
        promt_3,
        use_title=False,
        use_question=True,
    ):

        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        self.answers = []
        self.questions = []
        self.titles = []

        self.max_answer_length = 0

        for _, row in df.iterrows():

            prep_text = ""

            prep_text += promt_1
            if use_title:
                prep_text += row.Q_Title
            prep_text += promt_2
            if use_question:
                prep_text += row.Q_Body
            prep_text += promt_3

            question_len = len(tokenizer(prep_text)["input_ids"])

            prep_text += row.A_Body

            encoding_dict = tokenizer(
                prep_text
                # , truncation=True, max_length=128, padding="max_length"
            )

            self.input_ids.append(torch.tensor(encoding_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encoding_dict["attention_mask"]))
            self.labels.append(torch.tensor(encoding_dict["input_ids"]))
            self.labels[-1][:question_len] = -100

            self.answers.append(row.A_Body)
            self.questions.append(row.Q_Body)
            self.titles.append(row.Q_Title)

            answer_tokens_length = len(tokenizer.encode(row.A_Body))
            if self.max_answer_length <= answer_tokens_length:
                self.max_answer_length = answer_tokens_length

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attn_masks[idx],
            self.labels[idx],
            self.answers[idx],
            self.questions[idx],
            self.titles[idx],
        )


def collate_batch(examples, tokenizer, input_type="input_ids"):

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)

    if input_type == "input_ids":
        result = examples[0].new_full(
            [len(examples), max_length], tokenizer.pad_token_id
        )
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
    elif input_type == "attention_mask":
        result = examples[0].new_full([len(examples), max_length], 0)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
    return result


class Evaluator:
    def __init__(self, artefact, model, tokenizer) -> None:

        if artefact is not None:
            self.artefact = artefact
            self.text_table = wandb.Table(
                columns=[
                    "title",
                    "question",
                    "generated_answer",
                    "original_answer",
                    "bert_precision",
                    "bert_recall",
                    "bert_f1",
                    "rouge_score",
                    "bleu_score",
                ]
            )

        self.model = model
        self.tokenizer = tokenizer

        self.device = torch.device("cuda") if torch.cuda.is_available else "cpu"

        rouge = load("rouge")
        bertscore = load("bertscore")

        self.metrics = {
            "rouge": rouge.compute,
            "bertscore": bertscore.compute,
            "bleu": sentence_bleu,
        }

    def _generate_answer(
        self,
        question,
        title,
        promt_1,
        promt_2,
        promt_3,
        use_title=False,
        use_question=True,
        temp=0,
        max_answer_length=500,
    ):
        self.model.eval()

        text_to_answer = promt_1
        if use_title:
            text_to_answer += title
        text_to_answer += promt_2
        if use_question:
            text_to_answer += question
        text_to_answer += promt_3

        question_length = len(text_to_answer)

        enc_text_to_answer = self.tokenizer(
            text_to_answer, return_tensors="pt"
        ).input_ids.to(self.device)

        generated_output = self.model.generate(
            enc_text_to_answer,
            do_sample=False,
            top_k=50,
            top_p=0.9,
            temperature=temp,
            num_return_sequences=0,
            no_repeat_ngram_size=2,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_answer_length,
        ).to("cpu")

        del enc_text_to_answer

        generated_q_a = self.tokenizer.decode(
            generated_output[0], skip_special_tokens=True
        )

        generated_a = generated_q_a[question_length:]

        return generated_a

    def evaluate(
        self,
        test_dataset,
        promt_1,
        promt_2,
        promt_3,
        use_title=False,
        use_question=True,
        temp=0,
    ):
        self.max_answer_length = test_dataset.max_answer_length

        bleu_scores = []
        rouge_scores = []
        bert_precisions = []
        bert_recalls = []
        bert_f1s = []

        self.model.eval()

        for _, _, _, answer, question, title in tqdm(test_dataset):

            generated_answer = self._generate_answer(
                question,
                title,
                promt_1,
                promt_2,
                promt_3,
                use_title,
                use_question,
                temp,
                self.max_answer_length,
            )

            rouge_score = self.metrics["rouge"](
                predictions=[generated_answer], references=[answer]
            )["rouge1"].mid.fmeasure

            bert_score = self.metrics["bertscore"](
                predictions=[generated_answer], references=[answer], lang="en"
            )

            bleu_score = self.metrics["bleu"](
                answer, generated_answer, weights=(1, 0, 0, 0)
            )

            bleu_scores.append(bleu_score)
            rouge_scores.append(rouge_score)
            bert_precisions.append(bert_score["precision"][0])
            bert_recalls.append(bert_score["recall"][0])
            bert_f1s.append(bert_score["f1"][0])

            self.text_table.add_data(
                title,
                question,
                generated_answer,
                answer,
                bert_score["precision"][0],
                bert_score["recall"][0],
                bert_score["f1"][0],
                rouge_score,
                bleu_score,
            )

        return (
            self.text_table,
            np.mean(bleu_scores),
            np.mean(rouge_scores),
            np.mean(bert_precisions),
            np.mean(bert_recalls),
            np.mean(bert_f1s),
        )
