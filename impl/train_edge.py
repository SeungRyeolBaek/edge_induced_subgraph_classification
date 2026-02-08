# impl/train_edge.py
import torch

def _unpack_batch(batch):
    """
    Edge-dataset batch expected formats:

    Case A) z 없음:
      (x, edge_index, edge_weight, subG_node, y)

    Case B) z 있음 (여기서는 subG_edge를 z로 넣는 구조):
      (x, edge_index, edge_weight, subG_node, subG_edge, y)

    Case C) (subG_edge, node_z) 같이 튜플을 z로 줄 수도 있음:
      (x, edge_index, edge_weight, subG_node, z, y)
      where z is tensor (...,2) or tuple(subG_edge, node_z)

    반환:
      args_for_model(list), y
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError("Batch must be a list/tuple.")

    if len(batch) < 5:
        raise ValueError(f"Unexpected batch length: {len(batch)} (need >= 5)")

    y = batch[-1]
    core = batch[:-1]

    # 최소 (x, edge_index, edge_weight, subG_node)
    x = core[0]
    edge_index = core[1]
    edge_weight = core[2]
    subG_node = core[3]

    # z가 있는 경우만 추가
    if len(core) >= 5:
        z = core[4]
        return [x, edge_index, edge_weight, subG_node, z], y
    else:
        return [x, edge_index, edge_weight, subG_node], y


def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()

        args, y = _unpack_batch(batch)

        # 항상 id는 keyword로만 넣고, positional에는 절대 안 섞이게 고정
        pred = model(*args, id=0)

        loss = loss_fn(pred, y)
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()

    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        args, y = _unpack_batch(batch)

        # test에서는 id를 안 주면 forward default(id=0)로 감
        pred = model(*args)

        preds.append(pred)
        ys.append(y)

    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
