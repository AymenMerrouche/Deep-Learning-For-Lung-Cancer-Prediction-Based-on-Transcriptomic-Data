from utils import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_auc_score



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fit(checkpoint, criterion ,train_loader, val_loader, epochs, clip=float('inf'),entropy_param=0., writer=None, embedding_computer=None):
    """Full training loop"""

    print("Training on", 'GPU' if device.type == 'cuda' else 'CPU', '\n')
    net, optimizer = checkpoint.model, checkpoint.optimizer
    min_loss = float('inf')
    iteration = 1
    best_auc = float('-inf')
    def train_epoch():
        """
        Returns:
            The epoch loss
        """
        nonlocal iteration
        epoch_loss = 0.
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', dynamic_ncols=True,  position=0, leave=True)  # progress bar
        net.train()
        for (i, batch) in enumerate(pbar):
            images, labels = batch
            images, labels = images.to(device), labels.to(device).long()
            if embedding_computer is not None:
                with torch.no_grad():
                    mu, log_sigma_squared = embedding_computer.encoder(images)
                    images = embedding_computer.get_codes(mu, log_sigma_squared)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels) #.unsqueeze(-1)
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4e}')
            if writer:
                writer.add_scalar('Iteration_loss', loss.item(), iteration)
                
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=clip)
            if writer:
                writer.add_scalar('Total_norm_of_parameters_gradients', total_norm, iteration)
            optimizer.step()
            iteration += 1
            del images
            del labels
            torch.cuda.empty_cache()
        epoch_loss /= len(train_loader)
        return epoch_loss
    
    def categorical_accuracy(preds, y):
        """
        Returns accuracy per batch
        """
        y = y.cpu()
        preds = preds.cpu()
        max_preds = preds.argmax(dim=1, keepdim=True)  
        correct = max_preds.squeeze(1).eq(y) # 
        return correct.sum() / torch.FloatTensor([y.shape[0]])
    
    def evaluate_epoch(loader, role='Val'):
        """
        Args:
            loader (torch.utils.data.DataLoader): either the train of validation loader
            role (str): either 'Val' or 'Train'
        Returns:
            Tuple containing mean loss and accuracy
        """
        net.eval()
        correct = 0
        mean_loss = 0.
        predictions_stacked = []
        lebels_stacked = []
        outputs_stacked = []
        with torch.no_grad():
            for batch in loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device).long()
                if embedding_computer is not None:
                    with torch.no_grad():
                        mu, log_sigma_squared = embedding_computer.encoder(images)
                        images = embedding_computer.get_codes(mu, log_sigma_squared)
                output = net(images)
                loss = criterion(output, labels) #.unsqueeze(-1)
                mean_loss += loss.item()
                acc = categorical_accuracy(output, labels)
                correct += acc.item()
                preds = output.argmax(dim=1, keepdim=True)
                predictions_stacked.append(preds)
                outputs_stacked.append(output[:,1])
                lebels_stacked.append(labels)
                
        pred = torch.cat(predictions_stacked, 0).cpu().numpy()
        target = torch.cat(lebels_stacked, 0).cpu().numpy()
        probs = torch.cat(outputs_stacked, 0).cpu().numpy()
        target_names = ['No Cancer', 'Cancer']
        clf_report  = classification_report(target, pred, target_names=target_names)
        # auc
        auc = roc_auc_score(target, probs)
        return mean_loss / len(loader), correct / len(loader), auc, clf_report

    begin_epoch = checkpoint.epoch

    for epoch in range(begin_epoch, epochs+1):
        train_epoch()
        loss_train, acc_train, auc_train, _ = evaluate_epoch(train_loader, 'Train')
        loss_test, acc_test, auc_test, clf_report_test =  evaluate_epoch(val_loader, 'Val')
        

        print(f"Epoch {epoch}/{epochs}, Train Loss: {loss_train:.4e}, Test Loss: {loss_test:.4f}")
        print(f"Epoch {epoch}/{epochs}, Train Accuracy: {acc_train*100:.2f}%, Test Accuracy: {acc_test*100:.2f}%")
        print(f"Epoch {epoch}/{epochs}, Train AUC: {auc_train*100:.2f}%, Test AUC: {auc_test*100:.2f}%")
        print("Classification Report on Val Set : ")
        print(clf_report_test)
        if writer:
            writer.add_scalars("Loss", {"Train": loss_train, "Test" : loss_test}, epoch)
            writer.add_scalars("Accuracy", {"Train": acc_train*100, "Test" : acc_test*100}, epoch)
            writer.add_scalars("AUC", {"Train": auc_train*100, "Test" : auc_test*100}, epoch)
        checkpoint.epoch += 1
        if auc_test > best_auc:
            best_auc = auc_test
            best_acc = acc_test
            checkpoint.save('_best')
        checkpoint.save()

    print("\nFinished.")
    print(f"Best validation AUC: {best_auc:.4e}")
    print(f"With accuracy: {best_acc}")