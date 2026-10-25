
import torch
import torch.nn as nn

from .lstm_cells import VanillaLSTMCell, WMCLSTMCell, CIFGLSTMCell


CELL_TYPES = {
    "vanilla": VanillaLSTMCell,
    "wmc": WMCLSTMCell,
    "cifg": CIFGLSTMCell,
}


class BahdanauAttention(nn.Module):
    """Attention additive de Bahdanau."""

    def __init__(self, hidden_size: int):
        super().__init__()

        self.encoder_projection = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.decoder_projection = nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, source_lengths):
        """
        decoder_hidden : [batch, hidden]
        encoder_outputs : [batch, source_length, hidden]
        source_lengths : [batch]
        """
        encoder_features = self.encoder_projection(encoder_outputs)
        decoder_features = self.decoder_projection(
            decoder_hidden
        ).unsqueeze(1)

        energies = self.score(
            torch.tanh(encoder_features + decoder_features)
        ).squeeze(-1)

        positions = torch.arange(
            encoder_outputs.size(1),
            device=encoder_outputs.device,
        ).unsqueeze(0)

        mask = positions >= source_lengths.to(
            encoder_outputs.device
        ).unsqueeze(1)

        energies = energies.masked_fill(mask, -1e9)

        attention_weights = torch.softmax(energies, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context, attention_weights


class Encoder(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        cell_type: str,
        pad_idx: int = 0,
    ):
        super().__init__()

        if cell_type not in CELL_TYPES:
            raise ValueError(f"Type de cellule inconnu : {cell_type}")

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            source_vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )
        self.cell = CELL_TYPES[cell_type](
            embedding_dim, hidden_size
        )

    def forward(self, source, source_lengths):
        """
        source : [batch, source_length]
        """
        batch_size, source_length = source.shape
        embedded = self.embedding(source)

        h = embedded.new_zeros(batch_size, self.hidden_size)
        c = embedded.new_zeros(batch_size, self.hidden_size)

        outputs = []

        for t in range(source_length):
            h_new, c_new = self.cell(embedded[:, t, :], h, c)

            # Ne pas modifier l'état lorsque la position est du padding.
            valid = (t < source_lengths.to(source.device)).unsqueeze(1)

            h = torch.where(valid, h_new, h)
            c = torch.where(valid, c_new, c)

            outputs.append(h)

        encoder_outputs = torch.stack(outputs, dim=1)

        return encoder_outputs, h, c


class Decoder(nn.Module):
    def __init__(
        self,
        target_vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        cell_type: str,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            target_vocab_size,
            embedding_dim,
            padding_idx=pad_idx,
        )

        self.cell = CELL_TYPES[cell_type](
            embedding_dim, hidden_size
        )

        self.attention = BahdanauAttention(hidden_size)

        self.output_projection = nn.Linear(
            hidden_size * 2, target_vocab_size
        )

    def forward_step(
        self,
        token,
        hidden,
        cell_state,
        encoder_outputs,
        source_lengths,
    ):
        """
        token : [batch]
        hidden, cell_state : [batch, hidden]
        """
        embedded = self.embedding(token)

        hidden, cell_state = self.cell(
            embedded, hidden, cell_state
        )

        context, attention_weights = self.attention(
            hidden, encoder_outputs, source_lengths
        )

        combined = torch.cat([hidden, context], dim=1)
        logits = self.output_projection(combined)

        return logits, hidden, cell_state, attention_weights


class Seq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        cell_type: str = "vanilla",
        pad_idx: int = 0,
        sos_idx: int = 2,
        eos_idx: int = 3,
    ):
        super().__init__()

        if cell_type not in CELL_TYPES:
            raise ValueError(f"Type de cellule inconnu : {cell_type}")

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.hidden_size = hidden_size

        self.encoder = Encoder(
            source_vocab_size,
            embedding_dim,
            hidden_size,
            cell_type,
            pad_idx,
        )

        self.decoder = Decoder(
            target_vocab_size,
            embedding_dim,
            hidden_size,
            cell_type,
            pad_idx,
        )

    def forward(
        self,
        source,
        target,
        source_lengths=None,
        teacher_forcing_ratio=0.5,
    ):
        """
        source : [batch, source_length]
        target : [batch, target_length], avec SOS au début.

        Retour : logits [batch, target_length - 1, vocabulaire cible]
        """
        batch_size = source.size(0)

        if source_lengths is None:
            source_lengths = (source != self.pad_idx).sum(dim=1)

        encoder_outputs, hidden, cell_state = self.encoder(
            source, source_lengths
        )

        decoder_input = target[:, 0]
        outputs = []

        for t in range(1, target.size(1)):
            logits, hidden, cell_state, _ = self.decoder.forward_step(
                decoder_input,
                hidden,
                cell_state,
                encoder_outputs,
                source_lengths,
            )

            outputs.append(logits)

            if self.training and teacher_forcing_ratio > 0:
                use_teacher_forcing = (
                    torch.rand(batch_size, device=source.device)
                    < teacher_forcing_ratio
                )

                predicted_tokens = logits.argmax(dim=1)

                decoder_input = torch.where(
                    use_teacher_forcing,
                    target[:, t],
                    predicted_tokens,
                )
            else:
                decoder_input = logits.argmax(dim=1)

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def translate(
        self,
        source,
        source_lengths=None,
        max_length=50,
    ):
        """Génère une traduction par décodage glouton."""
        self.eval()

        if source_lengths is None:
            source_lengths = (source != self.pad_idx).sum(dim=1)

        encoder_outputs, hidden, cell_state = self.encoder(
            source, source_lengths
        )

        batch_size = source.size(0)

        decoder_input = torch.full(
            (batch_size,),
            self.sos_idx,
            dtype=torch.long,
            device=source.device,
        )

        generated = []
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=source.device
        )

        for _ in range(max_length):
            logits, hidden, cell_state, _ = self.decoder.forward_step(
                decoder_input,
                hidden,
                cell_state,
                encoder_outputs,
                source_lengths,
            )

            next_token = logits.argmax(dim=1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.eos_idx),
                next_token,
            )

            generated.append(next_token)
            finished |= next_token.eq(self.eos_idx)
            decoder_input = next_token

            if finished.all():
                break

        return torch.stack(generated, dim=1)

    def count_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
