import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import corpus_bleu
import random
import time
import sys
import matplotlib.pyplot as plt
import os
import numpy as np


# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
class Config:
    # File Paths
    TRAIN_EN_PATH = "Data/Train/train.en"
    TRAIN_FR_PATH = "Data/Train/train.fr"
    VAL_EN_PATH = "Data/Value/val.en"
    VAL_FR_PATH = "Data/Value/val.fr"

    # Thư mục lưu biểu đồ (Cho Model không Attention)
    GRAPH_SAVE_DIR = "Graph/Model"

    TEST_EN_PATH = "Data/Test/test_2016_flickr.en"
    TEST_FR_PATH = "Data/Test/test_2016_flickr.fr"

    # Model Hyperparameters (No Attention)
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Training Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    N_EPOCHS = 15
    CLIP = 1
    PATIENCE = 3

    NUM_RUNS = 10

    # Special Tokens
    SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_device()

# ==========================================
# 2. XỬ LÝ DỮ LIỆU (DATA PROCESSING)
# ==========================================
print("\n--- Đang xử lý dữ liệu ---")

try:
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")
except OSError:
    print("Vui lòng cài đặt spacy models.")
    sys.exit()


def read_data(path_en, path_fr):
    with open(path_en, "r", encoding="utf-8") as f:
        data_en = [line.strip() for line in f]
    with open(path_fr, "r", encoding="utf-8") as f:
        data_fr = [line.strip() for line in f]
    return list(zip(data_en, data_fr))


def yield_tokens(data_iterator, tokenizer, index):
    for data_sample in data_iterator:
        yield tokenizer(data_sample[index])


train_data = read_data(Config.TRAIN_EN_PATH, Config.TRAIN_FR_PATH)
val_data = read_data(Config.VAL_EN_PATH, Config.VAL_FR_PATH)
test_data = read_data(Config.TEST_EN_PATH, Config.TEST_FR_PATH)

vocab_en = build_vocab_from_iterator(yield_tokens(train_data, en_tokenizer, 0), min_freq=1,
                                     specials=Config.SPECIAL_TOKENS, special_first=True)
vocab_fr = build_vocab_from_iterator(yield_tokens(train_data, fr_tokenizer, 1), min_freq=1,
                                     specials=Config.SPECIAL_TOKENS, special_first=True)

vocab_en.set_default_index(vocab_en['<unk>'])
vocab_fr.set_default_index(vocab_fr['<unk>'])


def text_pipeline(text, tokenizer, vocab):
    tokens = tokenizer(text)
    indices = [vocab['<sos>']] + vocab(tokens) + [vocab['<eos>']]
    return torch.tensor(indices, dtype=torch.long)


def collate_batch(batch):
    processed_batch = []
    for src_text, trg_text in batch:
        src_tensor = text_pipeline(src_text, en_tokenizer, vocab_en)
        trg_tensor = text_pipeline(trg_text, fr_tokenizer, vocab_fr)
        processed_batch.append((src_tensor, trg_tensor, len(src_tensor)))
    processed_batch.sort(key=lambda x: x[2], reverse=True)
    src_list, trg_list, src_lens = zip(*processed_batch)
    src_batch = pad_sequence(src_list, padding_value=vocab_en['<pad>'])
    trg_batch = pad_sequence(trg_list, padding_value=vocab_fr['<pad>'])
    src_lens = torch.tensor(src_lens, dtype=torch.int64)
    return src_batch, trg_batch, src_lens


train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
if len(val_data) == 0:
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = torch.utils.data.random_split(train_data, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch)
else:
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, collate_fn=collate_batch)


# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH (NO ATTENTION)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)
        _, (hidden, cell) = self.rnn(packed_embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src, src_len)
        input_token = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input_token = trg[t] if random.random() < teacher_forcing_ratio else top1
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# ==========================================
# 4. TRAINING & EVAL UTILITIES
# ==========================================
def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg, src_len in iterator:
        src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg, src_len)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg, src_len in iterator:
            src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
            output = model(src, trg, src_len, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def translate_sentence(sentence, model, max_len=50):
    model.eval()
    tokens = [token for token in en_tokenizer(sentence)]
    indices = [vocab_en['<sos>']] + [vocab_en[t] for t in tokens] + [vocab_en['<eos>']]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(DEVICE)
    src_len = torch.LongTensor([len(indices)]).to(DEVICE)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    trg_indices = [vocab_fr['<sos>']]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(DEVICE)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        if pred_token == vocab_fr['<eos>']:
            break
    trg_tokens = [vocab_fr.lookup_token(i) for i in trg_indices]
    return " ".join(trg_tokens[1:-1])


def calculate_bleu_on_test_set(model, test_en_path, test_fr_path):
    print(f"   [Evaluating BLEU...]")
    model.eval()
    with open(test_en_path, 'r', encoding='utf-8') as f:
        test_en = [line.strip() for line in f]
    with open(test_fr_path, 'r', encoding='utf-8') as f:
        test_fr = [line.strip() for line in f]
    predictions = []
    references = []
    for i in range(len(test_en)):
        src = test_en[i]
        trg = test_fr[i]
        pred_sent = translate_sentence(src, model)
        predictions.append(fr_tokenizer(pred_sent))
        references.append([fr_tokenizer(trg)])
    score = corpus_bleu(references, predictions)
    return score


def translate_custom_sentences(model, sentence_pairs):
    print(f"\n   --- Custom Translations ---")
    model.eval()
    for i, (src, ref) in enumerate(sentence_pairs):
        pred = translate_sentence(src, model)
        print(f"   In: {src}")
        print(f"   Out: {pred}")


def draw_loss_chart(train_losses, val_losses, run_id, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(val_losses, label='Validation Loss', marker='o', color='red')
    plt.title(f'Training & Validation Loss - RUN {run_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    file_name = f"loss_chart_run_{run_id}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"   📊 Đã lưu biểu đồ tại: {save_path}")


# ==========================================
# 5. MAIN EXECUTION (VÒNG LẶP 10 LẦN)
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(Config.GRAPH_SAVE_DIR):
        os.makedirs(Config.GRAPH_SAVE_DIR)
        print(f"Đã tạo thư mục: {Config.GRAPH_SAVE_DIR}")

    print(f"\n🚀 BẮT ĐẦU CHẠY THỰC NGHIỆM {Config.NUM_RUNS} LẦN ĐỘC LẬP (NO ATTENTION)")

    # List để lưu kết quả BLEU của từng Run
    all_runs_bleu = []

    # Danh sách câu test nhanh
    my_sentences = [
        ("A black dog is running on the grass.", "Un chien noir court sur l'herbe."),
        ("Two men are playing soccer in the park.", "Deux hommes jouent au football dans le parc."),
        ("A little girl is eating an apple.", "Une petite fille mange une pomme.")
    ]

    for run_i in range(1, Config.NUM_RUNS + 1):
        print(f"\n{'=' * 20} RUN {run_i}/{Config.NUM_RUNS} {'=' * 20}")

        # --- KHỞI TẠO MODEL MỚI TINH ---
        enc = Encoder(len(vocab_en), Config.ENC_EMB_DIM, Config.HID_DIM, Config.N_LAYERS, Config.ENC_DROPOUT)
        dec = Decoder(len(vocab_fr), Config.DEC_EMB_DIM, Config.HID_DIM, Config.N_LAYERS, Config.DEC_DROPOUT)
        model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab_fr['<pad>'])

        # Đặt tên file model riêng cho từng run
        current_model_path = f"best_model_no_attn_run_{run_i}.pt"

        best_valid_loss = float('inf')
        no_improve_epoch = 0
        train_history = []
        valid_history = []

        # --- TRAINING LOOP ---
        for epoch in range(Config.N_EPOCHS):
            start_time = time.time()

            train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.CLIP)
            valid_loss = evaluate_epoch(model, val_loader, criterion)

            train_history.append(train_loss)
            valid_history.append(valid_loss)

            scheduler.step(valid_loss)

            end_time = time.time()
            mins, secs = divmod(end_time - start_time, 60)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), current_model_path)
                no_improve_epoch = 0
                save_msg = "✅ Save Best"
            else:
                no_improve_epoch += 1
                save_msg = f"⚠️ No improve ({no_improve_epoch}/{Config.PATIENCE})"

            print(
                f'   Ep {epoch + 1:02} | {int(mins)}m {int(secs)}s | Tr: {train_loss:.3f} | Val: {valid_loss:.3f} | {save_msg}')

            if no_improve_epoch >= Config.PATIENCE:
                print("   🛑 Early Stopping!")
                break

        # Lưu biểu đồ
        draw_loss_chart(train_history, valid_history, run_i, Config.GRAPH_SAVE_DIR)

        # --- ĐÁNH GIÁ (EVALUATION) CHO RUN NÀY ---
        print(f"\n   🔎 Đang đánh giá Run {run_i}...")
        model.load_state_dict(torch.load(current_model_path, map_location=DEVICE))

        # 1. Tính BLEU trên tập test
        bleu_score = calculate_bleu_on_test_set(model, Config.TEST_EN_PATH, Config.TEST_FR_PATH)
        all_runs_bleu.append(bleu_score)
        print(f"   🏆 BLEU Score (Run {run_i}): {bleu_score * 100:.2f}")

        # 2. Dịch thử vài câu
        translate_custom_sentences(model, my_sentences)

        # Dọn dẹp bộ nhớ GPU
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # --- TỔNG KẾT ---
    print(f"\n{'=' * 40}")
    print(f"KẾT QUẢ TỔNG HỢP SAU {Config.NUM_RUNS} LẦN CHẠY (NO ATTENTION)")
    print(f"{'=' * 40}")
    print(f"Chi tiết BLEU từng run: {[round(b * 100, 2) for b in all_runs_bleu]}")
    print(f"Trung bình cộng (Mean BLEU): {np.mean(all_runs_bleu) * 100:.2f}")
    print(f"Độ lệch chuẩn (Std Dev): {np.std(all_runs_bleu) * 100:.2f}")
    print(f"Cao nhất (Max): {np.max(all_runs_bleu) * 100:.2f}")
    print(f"Thấp nhất (Min): {np.min(all_runs_bleu) * 100:.2f}")