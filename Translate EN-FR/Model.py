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
from functools import partial

# ==========================================
# 1. CÀI ĐẶT THAM SỐ TOÀN CỤC (GLOBAL VARS)
# ==========================================
# --- Đường dẫn file dữ liệu (Cần tồn tại trước) ---
TRAIN_EN_PATH = "Data/Train/train.en"
TRAIN_FR_PATH = "Data/Train/train.fr"
VAL_EN_PATH = "Data/Value/val.en"
VAL_FR_PATH = "Data/Value/val.fr"
TEST_EN_PATH = "Data/Test/test_2016_flickr.en"
TEST_FR_PATH = "Data/Test/test_2016_flickr.fr"

# --- Cấu hình thư mục đầu ra (Tự động tạo) ---
OUTPUT_DIR = "outputs"  # Tất cả kết quả sẽ lưu vào đây
MODEL_NAME = "best_model.pth"
GRAPH_NAME = "loss_chart.png"

# --- Tham số Model ---
ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# --- Tham số Huấn luyện ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
N_EPOCHS = 15
CLIP = 1
PATIENCE = 3
TEACHER_FORCING_RATIO = 0.5

# --- Token đặc biệt ---
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']


# --- Thiết bị ---
def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


DEVICE = get_device()
print(f"🔹 Đang sử dụng thiết bị: {DEVICE}")


# ==========================================
# 2. XỬ LÝ DỮ LIỆU
# ==========================================
class TextProcessor:
    def __init__(self):
        try:
            self.en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
            self.fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")
        except OSError:
            print("❌ Lỗi: Thiếu thư viện spacy. Chạy lệnh: python -m spacy download en_core_web_sm")
            sys.exit()
        self.vocab_en = None
        self.vocab_fr = None

    def build_vocab(self, train_data):
        print("🔹 Đang xây dựng bộ từ điển...")

        def yield_tokens(data, tokenizer, idx):
            for sample in data:
                yield tokenizer(sample[idx])

        self.vocab_en = build_vocab_from_iterator(
            yield_tokens(train_data, self.en_tokenizer, 0),
            min_freq=1, specials=SPECIAL_TOKENS, special_first=True
        )
        self.vocab_fr = build_vocab_from_iterator(
            yield_tokens(train_data, self.fr_tokenizer, 1),
            min_freq=1, specials=SPECIAL_TOKENS, special_first=True
        )
        self.vocab_en.set_default_index(UNK_IDX)
        self.vocab_fr.set_default_index(UNK_IDX)
        print(f"   Size EN: {len(self.vocab_en)} | Size FR: {len(self.vocab_fr)}")

    def text_pipeline(self, text, tokenizer, vocab):
        tokens = tokenizer(text)
        indices = [SOS_IDX] + vocab(tokens) + [EOS_IDX]
        return torch.tensor(indices, dtype=torch.long)


def read_data(path_en, path_fr):
    with open(path_en, "r", encoding="utf-8") as f: en = [l.strip() for l in f]
    with open(path_fr, "r", encoding="utf-8") as f: fr = [l.strip() for l in f]
    return list(zip(en, fr))


def collate_fn(batch, processor):
    batch.sort(key=lambda x: len(processor.text_pipeline(x[0], processor.en_tokenizer, processor.vocab_en)),
               reverse=True)
    src_list, trg_list, src_lens = [], [], []
    for src_text, trg_text in batch:
        src_tensor = processor.text_pipeline(src_text, processor.en_tokenizer, processor.vocab_en)
        trg_tensor = processor.text_pipeline(trg_text, processor.fr_tokenizer, processor.vocab_fr)
        src_list.append(src_tensor)
        trg_list.append(trg_tensor)
        src_lens.append(len(src_tensor))
    src_batch = pad_sequence(src_list, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_list, padding_value=PAD_IDX)
    return src_batch, trg_batch, torch.tensor(src_lens, dtype=torch.int64)


# ==========================================
# 3. MÔ HÌNH (SEQ2SEQ)
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

    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
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
# 4. HUẤN LUYỆN & EVAL
# ==========================================
def train_step(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg, src_len in iterator:
        src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg, src_len, TEACHER_FORCING_RATIO)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate_step(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg, src_len in iterator:
            src, trg, src_len = src.to(DEVICE), trg.to(DEVICE), src_len.to(DEVICE)
            output = model(src, trg, src_len, 0)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def translate_sentence(sentence, model, processor, max_len=50):
    model.eval()
    tokens = processor.en_tokenizer(sentence)
    indices = [SOS_IDX] + processor.vocab_en(tokens) + [EOS_IDX]
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(DEVICE)
    src_len = torch.LongTensor([len(indices)]).to(DEVICE)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
    trg_indices = [SOS_IDX]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(DEVICE)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        if pred_token == EOS_IDX: break
    trg_tokens = [processor.vocab_fr.lookup_token(i) for i in trg_indices]
    return " ".join(trg_tokens[1:-1])


# ==========================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ==========================================
if __name__ == "__main__":
    # --- TỰ ĐỘNG TẠO THƯ MỤC LƯU ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"🔹 Đã tạo thư mục lưu trữ: {OUTPUT_DIR}")

    # Định nghĩa đường dẫn file đầy đủ
    full_model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    full_graph_path = os.path.join(OUTPUT_DIR, GRAPH_NAME)

    # 1. Load Data
    processor = TextProcessor()
    raw_train = read_data(TRAIN_EN_PATH, TRAIN_FR_PATH)
    raw_val = read_data(VAL_EN_PATH, VAL_FR_PATH)
    processor.build_vocab(raw_train)

    train_loader = DataLoader(raw_train, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, processor=processor),
                              shuffle=True)
    val_loader = DataLoader(raw_val, batch_size=BATCH_SIZE, collate_fn=partial(collate_fn, processor=processor))

    # 2. Init Model
    enc = Encoder(len(processor.vocab_en), ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(len(processor.vocab_fr), DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 3. Training Loop
    print(f"\n🚀 BẮT ĐẦU TRAINING ({N_EPOCHS} Epochs)...")
    best_valid_loss = float('inf')
    no_improve = 0
    train_history, valid_history = [], []

    for epoch in range(N_EPOCHS):
        start_t = time.time()

        train_loss = train_step(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate_step(model, val_loader, criterion)

        train_history.append(train_loss)
        valid_history.append(valid_loss)
        scheduler.step(valid_loss)

        mins, secs = divmod(time.time() - start_t, 60)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # SỬA LỖI: Dùng biến full_model_path đã định nghĩa
            torch.save(model.state_dict(), full_model_path)
            no_improve = 0
            save_msg = "✅ Saved Best"
        else:
            no_improve += 1
            save_msg = f"⚠️ ({no_improve}/{PATIENCE})"

        print(
            f"Ep {epoch + 1:02} | {int(mins)}m {int(secs)}s | Tr: {train_loss:.3f} | Val: {valid_loss:.3f} | {save_msg}")

        if no_improve >= PATIENCE:
            print("🛑 Early Stopping!")
            break

    # 4. Vẽ biểu đồ (SỬA LỖI ĐƯỜNG DẪN)
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Train Loss')
    plt.plot(valid_history, label='Val Loss')
    plt.legend()
    plt.title('Training Result')
    plt.grid(True)
    plt.savefig(full_graph_path)  # Lưu vào outputs/loss_chart.png
    print(f"📊 Đã lưu biểu đồ tại: {full_graph_path}")

    # 5. Đánh giá (Test)
    print("\n🔎 ĐÁNH GIÁ TRÊN TẬP TEST & DỊCH THỬ")
    # SỬA LỖI: Truyền đúng đường dẫn file model vào hàm load
    if os.path.exists(full_model_path):
        model.load_state_dict(torch.load(full_model_path, map_location=DEVICE))
    else:
        print("⚠️ Cảnh báo: Không tìm thấy file model để load!")

    new_test_sentences = [
        ("The cat sleeps on the sofa.", "Le chat dort sur le canapé."),
        ("I love learning new languages every day.", "J'adore apprendre de nouvelles langues chaque jour."),
        ("Where is the nearest train station?", "Où est la gare la plus proche ?"),
        ("She bought a beautiful red dress yesterday.", "Elle a acheté une belle robe rouge hier."),
        ("Can you help me with my homework please?", "Peux-tu m'aider avec mes devoirs s'il te plaît ?")
    ]

    print(f"{'=' * 50}")
    print("DỊCH THỬ 5 CÂU MỚI:")
    for i, (src, ref) in enumerate(new_test_sentences):
        pred = translate_sentence(src, model, processor)
        print(f"#{i + 1} In:  {src}")
        print(f"   Out: {pred}")
        print(f"   Ref: {ref}\n")
    print(f"{'=' * 50}")

    try:
        with open(TEST_EN_PATH, 'r', encoding='utf-8') as f:
            src_test = [l.strip() for l in f]
        with open(TEST_FR_PATH, 'r', encoding='utf-8') as f:
            trg_test = [l.strip() for l in f]
        preds, refs = [], []
        for s, t in zip(src_test, trg_test):
            preds.append(processor.fr_tokenizer(translate_sentence(s, model, processor)))
            refs.append([processor.fr_tokenizer(t)])
        score = corpus_bleu(refs, preds)
        print(f"🏆 Final BLEU Score on Test Set: {score * 100:.2f}")
    except Exception as e:
        print(f"⚠️ Không thể tính BLEU (Check file path): {e}")