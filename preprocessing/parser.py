import pandas as pd


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    messages = []
    msg_id = 1

    for conv_id, row in enumerate(df.itertuples(index=False), start=1):
        conv_text = str(row[0])
        lines = conv_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("User 1:"):
                speaker = "User 1"
                text = line.replace("User 1:", "").strip()

            elif line.startswith("User 2:"):
                speaker = "User 2"
                text = line.replace("User 2:", "").strip()

            else:
                continue

            messages.append({
                "msg_id": msg_id,
                "conv_id": conv_id,
                "speaker": speaker,
                "text": text
            })

            msg_id += 1

    return messages