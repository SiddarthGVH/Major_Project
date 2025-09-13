# toxicity_bert_audio.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import whisper
import argparse
import os

# ---------------------------
# Audio transcription
# ---------------------------
def transcribe_audio(audio_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    return result['text']

# ---------------------------
# Training function
# ---------------------------
def train_bert_model(outdir="tox_model_bert", num_samples=2000, epochs=1):
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
    from datasets import load_dataset

    print("📥 Loading dataset...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("civil_comments", split=f"train[:{num_samples}]")
    dataset = dataset.map(lambda x: {'labels': int(x['toxicity'] > 0.5)})

    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_strategy="no",
        save_strategy="epoch",
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"Model and tokenizer saved to {outdir}")
    return model, tokenizer

# ---------------------------
# Prediction function
# ---------------------------
# ---------------------------
# Prediction function
# ---------------------------
def predict_toxicity(text, model_dir="tox_model_bert"):
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    import torch.nn.functional as F
    import torch

    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    # probs[0] -> non-toxic, probs[1] -> toxic
    toxicity_score = float(probs[1])
    non_toxic_score = float(probs[0])
    prediction = "TOXIC" if toxicity_score >= 0.5 else "NOT TOXIC"

    return prediction, toxicity_score, non_toxic_score

# ---------------------------
# Main function
# ---------------------------
# ---------------------------
# Main function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--train_model", action="store_true", help="Train BERT model")
    parser.add_argument("--model_dir", type=str, default="tox_model_bert", help="Model directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()

    # If no audio and no train flag → interactive mode
    if not args.audio and not args.train_model:
        print("✨ Interactive mode ✨")
        choice = input("Do you want to (1) Train model or (2) Analyze audio? Enter 1/2: ").strip()

        if choice == "1":
            train_bert_model(outdir=args.model_dir, epochs=args.epochs)
            return
        elif choice == "2":
            args.audio = input("🎙️ Enter the path to the audio file: ").strip()
        else:
            print("❌ Invalid choice, exiting.")
            return

    # Train model if requested
    if args.train_model:
        train_bert_model(outdir=args.model_dir, epochs=args.epochs)

    # Analyze audio if provided
    if args.audio:
        if not os.path.exists(args.model_dir):
            print("⚠️ No trained model found. Training a quick model now...")
            train_bert_model(outdir=args.model_dir, epochs=args.epochs)

        text = transcribe_audio(args.audio)
        print("📝 Transcribed Text:", text)

        prediction, tox_score, non_tox_score = predict_toxicity(text, model_dir=args.model_dir)
        print(f"Prediction: {prediction}")
        print(f"Toxicity Score: {tox_score:.4f}")
        print(f"Non-Toxic Score: {non_tox_score:.4f}")

# ---------------------------
if __name__ == "__main__":
    main()
