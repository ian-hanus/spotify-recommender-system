from model_architecture import Autoencoder
import torch

def train_model(model: Autoencoder, dataloader, num_epochs=10):
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            labels = batch
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(dataloader)
        print('Epoch loss: ' + (epoch_loss))

    return model