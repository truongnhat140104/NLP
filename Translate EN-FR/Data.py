import torch
from spacy import tokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import random # Dùng cho Teacher Forcing
import torch.nn as nn
import torch.optim as optim
import time
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu

en_tokenizer = get_tokenizer("spacy",language="en_core_web_sm")
fr_tokenizer = get_tokenizer("spacy",language="fr_core_news_sm")

special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']

#train temp
with open("Data/Train/train.en", "r", encoding="utf-8") as f:
    train_data_en = [line.strip() for line in f]

# Đọc file tiếng Pháp
with open("Data/Train/train.fr", "r", encoding="utf-8") as f:
    train_data_fr = [line.strip() for line in f]

train_data = list(zip(train_data_en, train_data_fr))

def yield_en_tokens(data_iterator):
    for en_token in data_iterator:
        yield en_tokenizer(en_token) #Dua word_en vao yield

def yield_fr_tokens(data_iterator):
    for fr_token in data_iterator:
        yield fr_tokenizer(fr_token)

vocab_en = build_vocab_from_iterator(yield_en_tokens(train_data_en), min_freq=1, specials=special_tokens, special_first=True)
vocab_fr = build_vocab_from_iterator(yield_fr_tokens(train_data_fr), min_freq=1, specials=special_tokens, special_first=True)


#Xet default cho vocab
vocab_en.set_default_index(vocab_en['<unk>'])
vocab_fr.set_default_index(vocab_fr['<unk>'])

def text_transform(vocab, tokenizer, text):
    # String -> Token -> Index -> Thêm SOS/EOS -> Tensor
    tokens = tokenizer(text)
    indices = [vocab['<sos>']] + vocab(tokens) + [vocab['<eos>']]
    return torch.tensor(indices, dtype=torch.long)


def collate_batch(batch):
    processed_batch = []

    for src_text, trg_text in batch:
        src_tensor = text_transform(vocab_en, en_tokenizer, src_text) #source tensor
        trg_tensor = text_transform(vocab_fr, fr_tokenizer, trg_text) #target tensor
        # Lưu lại tuple: (src_tensor, trg_tensor, độ_dài_src)
        processed_batch.append((src_tensor, trg_tensor, len(src_tensor)))

    # --- QUAN TRỌNG: SẮP XẾP BATCH GIẢM DẦN THEO ĐỘ DÀI SRC ---
    # Để thỏa mãn yêu cầu enforce_sorted=True
    processed_batch.sort(key=lambda x: x[2], reverse=True)

    # Tách dữ liệu ra sau khi đã sắp xếp
    src_list = [x[0] for x in processed_batch]
    trg_list = [x[1] for x in processed_batch]
    src_lens = [x[2] for x in processed_batch]

    # Padding
    src_batch = pad_sequence(src_list, padding_value=vocab_en['<pad>'])
    trg_batch = pad_sequence(trg_list, padding_value=vocab_fr['<pad>'])

    # Chuyển length sang tensor int64 (CPU)
    src_lens = torch.tensor(src_lens, dtype=torch.int64)

    return src_batch, trg_batch, src_lens

BATCH_SIZE = 32 # Có thể chỉnh lên 64, 128

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    collate_fn=collate_batch,
    shuffle=True # Shuffle data gốc, nhưng trong mỗi batch sẽ tự sắp xếp lại
)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # 1. Embedding (Size 256-512)
        self.embedding = nn.Embedding(input_dim, emb_dim)

        # 2. LSTM (2 Layers, Hidden 512)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        # 3. Dropout (0.3-0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [src_len, batch_size]

        # Bước 1: Embedding & Dropout
        embedded = self.dropout(self.embedding(src))

        # Bước 2: Packing (Do bạn đã sắp xếp batch ở collate_fn nên để True)
        packed_embedded = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=True)

        # Bước 3: Đưa qua LSTM
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)

        # Trả về Context Vector (hidden, cell) để truyền sang Decoder
        return hidden, cell


# ==================================================
# 2. DECODER
# Input: (Token trước đó, Hidden trước đó, Cell trước đó)
# Output: (Dự đoán từ hiện tại, Hidden mới, Cell mới)
# ==================================================
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # 1. Embedding
        self.embedding = nn.Embedding(output_dim, emb_dim)

        # 2. LSTM (Cấu hình y hệt Encoder: 2 layers, 512 hidden)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        # 3. Linear: Biến Hidden state -> Xác suất từ vựng (Softmax sẽ được tính trong Loss function)
        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size] (chỉ là 1 từ tại thời điểm t)
        # hidden & cell: [n_layers, batch_size, hid_dim]

        # Thêm chiều sequence = 1 vào input -> [1, batch_size]
        input = input.unsqueeze(0)

        # Embedding
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]

        # Đưa qua LSTM
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # Dự đoán từ tiếp theo
        prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]

        return prediction, hidden, cell


# ==================================================
# 3. SEQ2SEQ (MÔ HÌNH TỔNG)
# Kết nối Encoder và Decoder + Xử lý Teacher Forcing
# ==================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # Kiểm tra Hidden size phải khớp nhau
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions của Encoder và Decoder phải giống nhau!"
        assert encoder.n_layers == decoder.n_layers, \
            "Số layer của Encoder và Decoder phải giống nhau!"

    def forward(self, src, trg, src_len, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size] (Câu đích dùng để huấn luyện)
        # teacher_forcing_ratio: Tỷ lệ dùng ground truth làm input tiếp theo (đề nghị 0.5)

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # Tensor chứa toàn bộ kết quả dự đoán
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # --- BƯỚC 1: ENCODER ---
        # Lấy context vector (hidden, cell) từ câu nguồn
        hidden, cell = self.encoder(src, src_len)

        # --- BƯỚC 2: DECODER LOOP ---
        # Input đầu tiên cho Decoder luôn là token <sos> (phần tử đầu của trg)
        input = trg[0, :]

        for t in range(1, trg_len):
            # Chạy Decoder cho 1 bước thời gian
            output, hidden, cell = self.decoder(input, hidden, cell)

            # Lưu kết quả dự đoán vào tensor outputs
            outputs[t] = output

            # --- TEACHER FORCING LOGIC ---
            # Quyết định input tiếp theo là gì?

            # 1. Lấy từ có xác suất cao nhất máy vừa đoán ra
            top1 = output.argmax(1)

            # 2. Tung xúc xắc: Nếu random < 0.5 -> Dùng đáp án thật (trg[t]). Ngược lại dùng top1.
            teacher_force = random.random() < teacher_forcing_ratio

            # Cập nhật input cho vòng lặp sau
            input = trg[t] if teacher_force else top1

        return outputs

# --- CẤU HÌNH THEO ĐỀ BÀI ---
INPUT_DIM = len(vocab_en)  # Kích thước từ điển Anh
OUTPUT_DIM = len(vocab_fr) # Kích thước từ điển Pháp

# Tham số từ bảng
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512       # Đề yêu cầu 512
N_LAYERS = 2        # Đề yêu cầu 2 layers
ENC_DROPOUT = 0.5   # Đề yêu cầu 0.3 - 0.5
DEC_DROPOUT = 0.5

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps') # Đây là GPU trên Mac
    print("Đang sử dụng Apple Metal (MPS) GPU acceleration")
else:
    device = torch.device('cpu')
    print("Đang sử dụng CPU")

# 1. Khởi tạo
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# --- TEST DỮ LIỆU ---
# Lấy batch từ DataLoader bạn đã viết trước đó
src_batch, trg_batch, src_lens = next(iter(train_loader))

# Chuyển sang device
src_batch = src_batch.to(device)
trg_batch = trg_batch.to(device)
src_lens = src_lens.to(device) # Lưu ý: Một số bản PyTorch cần src_lens ở CPU


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

# Cấu hình Optimizer và Loss theo yêu cầu
LEARNING_RATE = 0.001
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Bỏ qua token <pad> khi tính Loss
PAD_IDX = vocab_fr['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# --- CHUẨN BỊ DỮ LIỆU VALIDATION (Tách từ Train ra để làm Early Stopping) ---
# Tách 90% train, 10% validation
train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_set, val_set = torch.utils.data.random_split(train_data, [train_size, val_size])

# Tạo Loader mới
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_batch)


# ==================================================
# 5. TRAINING & EVALUATION LOOP
# ==================================================

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg, src_len) in enumerate(iterator):
        src, trg, src_len = src.to(device), trg.to(device), src_len.to(device)

        optimizer.zero_grad()

        # output: [trg_len, batch_size, output_dim]
        output = model(src, trg, src_len)

        # Bỏ token <sos> đầu tiên khi tính loss
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Cắt gradient để tránh bùng nổ gradient (Gradient Explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg, src_len) in enumerate(iterator):
            src, trg, src_len = src.to(device), trg.to(device), src_len.to(device)

            # Tắt teacher forcing khi evaluate (ratio = 0)
            output = model(src, trg, src_len, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# ==================================================
# 6. CHẠY HUẤN LUYỆN VỚI EARLY STOPPING
# ==================================================

N_EPOCHS = 20  # Yêu cầu: 10-20 epoch
CLIP = 1
patience = 3  # Early stopping patience
no_improve_epoch = 0
best_valid_loss = float('inf')

print(f"Bắt đầu huấn luyện trên thiết bị: {device}")

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    # Logic lưu Best Model & Early Stopping
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pt')
        no_improve_epoch = 0  # Reset đếm
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} (New Best!)')
    else:
        no_improve_epoch += 1
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        print(f'\tKhông cải thiện: {no_improve_epoch}/{patience}')

    # Dừng nếu val_loss không giảm sau 3 epoch
    if no_improve_epoch >= patience:
        print("Early Stopping! Dừng huấn luyện.")
        break

# Load lại model tốt nhất trước khi dự đoán
model.load_state_dict(torch.load('best_model.pt', weights_only=True))

def translate(sentence: str, max_len: int = 50):
    model.eval()

    # 1. Tokenize & chuyển thành index
    tokens = en_tokenizer(sentence)
    tokens = [token.lower() for token in tokens]  # Lowercase nếu cần

    # Thêm <sos> và <eos>
    indices = [vocab_en['<sos>']] + [vocab_en[token] for token in tokens] + [vocab_en['<eos>']]

    # Chuyển thành Tensor & thêm chiều batch
    src_tensor = torch.LongTensor(indices).unsqueeze(1).to(device)  # [src_len, 1]
    src_len = torch.LongTensor([len(indices)]).to(device)

    with torch.no_grad():
        # Encoder
        hidden, cell = model.encoder(src_tensor, src_len)

    # Khởi tạo input cho Decoder là <sos>
    trg_indexes = [vocab_fr['<sos>']]

    # Vòng lặp Decoder (Greedy Decoding)
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)  # Token vừa dự đoán

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        # Lấy token có xác suất cao nhất (Greedy)
        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        # Dừng nếu gặp <eos>
        if pred_token == vocab_fr['<eos>']:
            break

    # Chuyển từ index về từ vựng
    trg_tokens = [vocab_fr.lookup_token(i) for i in trg_indexes]

    # Loại bỏ <sos> và <eos> để trả về string sạch
    return " ".join(trg_tokens[1:-1])


# ==================================================
# 8. KIỂM TRA & ĐÁNH GIÁ (BLEU SCORE)
# ==================================================


print("\n--- TEST DỊCH THỬ ---")
# Ví dụ 1 câu tiếng Anh trong tập train để test
example_idx = 0
src_sample = train_data_en[example_idx]
trg_sample = train_data_fr[example_idx]

print(f"Src: {src_sample}")
print(f"Trg (Reference): {trg_sample}")
pred_sentence = translate(src_sample)
print(f"Pred: {pred_sentence}")

# Tính BLEU Score đơn giản cho câu này
# Reference cần là list of list tokens: [[t1, t2, ...]]
reference = [fr_tokenizer(trg_sample)]
candidate = fr_tokenizer(pred_sentence)
score = sentence_bleu(reference, candidate)
print(f"BLEU Score: {score:.4f}")