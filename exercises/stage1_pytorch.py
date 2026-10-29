"""Stage 1: offline PyTorch fundamentals. CPU by default; no downloads."""

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_and_gradient_demo():
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    weight = torch.ones(2, 2)
    bias = torch.tensor([10., 20.])
    result = x @ weight + bias
    assert result.shape == (3, 2)
    assert torch.equal(result[0], torch.tensor([13., 23.]))
    print('tensor:', x.shape, x.dtype, x.device, result[0].tolist())

    w = torch.tensor(2., requires_grad=True)
    (w - 1).square().backward()
    assert torch.isclose(w.grad, torch.tensor(2.))
    (w - 1).square().backward()  # A fresh graph, but the same leaf gradient.
    assert torch.isclose(w.grad, torch.tensor(4.))
    w.grad = None
    (w - 1).square().backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    assert torch.isclose(w, torch.tensor(1.8))
    print('gradient: 2 -> accumulated 4; after reset/update w=', w.item())


class PointsDataset(Dataset):
    def __init__(self, count, seed):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(count, 2, generator=generator)
        self.y = (self.x[:, 0] + 0.5 * self.x[:, 1] > 0).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def make_model():
    return nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        assert torch.isfinite(loss), 'Non-finite training loss'
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += len(y)
    return loss_sum / count, correct / count


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        count += len(y)
    return loss_sum / count, correct / count


def overfit_demo(device):
    seed_everything(7)
    dataset = PointsDataset(32, seed=77)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    before = model[0].weight.detach().clone()
    for _ in range(200):
        train_epoch(model, loader, optimizer, device)
    _, accuracy = evaluate(model, loader, device)
    assert accuracy >= 0.95, f'Overfit self-check failed: {accuracy}'
    assert not torch.equal(before, model[0].weight)
    print(f'overfit self-check: accuracy={accuracy:.3f}; parameters updated')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--out', type=Path, default=Path('stage1_output'))
    args = parser.parse_args()
    if args.epochs < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('Epoch counts must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use CPU or check the environment later')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    print('torch=', torch.__version__, 'device=', device)
    tensor_and_gradient_demo()
    overfit_demo(device)

    seed_everything(42)
    train_data, val_data = PointsDataset(512, 100), PointsDataset(256, 200)
    shuffle_rng = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                              generator=shuffle_rng, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    x, y = next(iter(val_loader))
    assert x.shape == (64, 2) and y.shape == (64,)
    assert x.dtype == torch.float32 and y.dtype == torch.int64
    model = make_model().to(device)
    assert model(x.to(device)).shape == (64, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = dict(seed=42, train_seed=100, val_seed=200,
                  batch_size=32, lr=0.01, hidden=16)
    initial_val_loss, _ = evaluate(model, val_loader, device)
    start_epoch, history = 0, []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        assert checkpoint['config'] == config, 'Checkpoint configuration mismatch'
        assert checkpoint['device'] == args.device, 'Use the same device type to resume'
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch, history = checkpoint['epoch'], checkpoint['history']
        initial_val_loss = checkpoint['initial_val_loss']
        shuffle_rng.set_state(checkpoint['shuffle_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        print('resumed after epoch', start_epoch)

    end_epoch = min(args.epochs, args.stop_after or args.epochs)
    if start_epoch >= end_epoch:
        parser.error('No epochs remain; increase --epochs/remove --stop-after')
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, end_epoch):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(dict(epoch=epoch + 1, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss, val_acc=val_acc))
        print(f'epoch={epoch+1:02d} train_loss={train_loss:.4f} '
              f'train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch + 1, global_step=(epoch + 1) * len(train_loader),
                        config=config, device=args.device, history=history,
                        initial_val_loss=initial_val_loss,
                        torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                        shuffle_rng=shuffle_rng.get_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else []),
                   args.out / 'last.pt')

    with (args.out / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # Test a fresh model instance against a fixed input after loading the file.
    model.eval()
    with torch.no_grad():
        expected = model(x.to(device))
    loaded = make_model().to(device)
    saved = torch.load(args.out / 'last.pt', map_location='cpu', weights_only=True)
    loaded.load_state_dict(saved['model'])
    loaded.eval()
    with torch.no_grad():
        actual = loaded(x.to(device))
    torch.testing.assert_close(actual, expected)
    if end_epoch >= 20:
        assert val_acc >= 0.90, f'Validation self-check failed: {val_acc}'
        assert val_loss < initial_val_loss
    print('PASS: reload predictions match; history=', args.out / 'metrics.csv')


if __name__ == '__main__':
    main()
