import torch
import clip
from tqdm import tqdm
from utils import template, prompts, save_results
from dataloader import preprocess_sepcgram


def train_one_epoch(model, epoch, epochs, device, train_loader, loss_func, optimizer, preprocess, scheduler):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train')
    for _, (filenames, labels) in enumerate(loop): # shape(4, *, 12)
        imgs = preprocess_sepcgram(filenames, preprocess).to(device) # shape(4,3,224,224)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in labels]).to(device)
        
        logits_per_image, logits_per_text = model(imgs, text)
        
        labels = torch.LongTensor(range(len(labels))).to(device)
        loss_I = loss_func(logits_per_image, labels)
        loss_T = loss_func(logits_per_text, labels)
        loss = (loss_I + loss_T) / 2
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = loss.item())
    scheduler.step()
    print("[%d/%d] epoch's total loss = %f" % (epoch + 1, epochs, total_loss))
    save_results('res/results.csv', '%d, %12.6f\n' % (epoch + 1, total_loss))


def evaluate(model, device, eval_loader, loss_func, preprocess):
    model.eval()
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Evaluating...")
    loop = tqdm(eval_loader, desc='Evaluation')
    for i, (filenames, labels) in enumerate(loop): # shape(4, *, 12)
        imgs = preprocess_sepcgram(filenames, preprocess).to(device) # shape(4,3,224,224)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in labels]).to(device)
        
        logits_per_image, logits_per_text = model(imgs, text)
        
        labels = torch.LongTensor(range(len(labels))).to(device)
        loss_I = loss_func(logits_per_image, labels)
        loss_T = loss_func(logits_per_text, labels)
        loss = (loss_I + loss_T) / 2
        total_loss += loss

        predict_idx = logits_per_image.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Evaluating [{i + 1}/{len(eval_loader)}]')
        loop.set_postfix(loss = loss.item())
    
    print("Evalution:")
    print("Total loss: {}".format(total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision: {}%".format(round(float(correct_nums) / total_nums, 6) * 100))


def train_one_cnn_epoch(model, epoch, epochs, device, train_loader, loss_func, optimizer, preprocess, scheduler):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train')
    for _, (filenames, labels) in enumerate(loop): # shape(4, *, 12)
        imgs = preprocess_sepcgram(filenames, preprocess).to(device) # shape(4,3,224,224)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in labels]).to(device)
        
        logits_per_image = model(imgs, text)
        
        labels = torch.tensor(labels, device=device) - 1
        loss = loss_func(logits_per_image, labels)
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = loss.item())
    scheduler.step()
    print("[%d/%d] epoch's total loss = %f" % (epoch + 1, epochs, total_loss))
    save_results('res/results.csv', '%d, %12.6f\n' % (epoch + 1, total_loss))


def evaluate_cnn(model, device, eval_loader, loss_func, preprocess):
    model.eval() # 增加dropout层后，精确度从30%提升到40%
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Evaluating...")
    loop = tqdm(eval_loader, desc='Evaluation')
    for i, (filenames, labels) in enumerate(loop): # shape(4, *, 12)
        imgs = preprocess_sepcgram(filenames, preprocess).to(device) # shape(4,3,224,224)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in labels]).to(device)
        
        logits_per_image = model(imgs, text)
        
        labels = torch.tensor(labels, device=device) - 1
        loss = loss_func(logits_per_image, labels)
        total_loss += loss

        predict_idx = logits_per_image.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Evaluating [{i + 1}/{len(eval_loader)}]')
        loop.set_postfix(loss = loss.item())
    
    print("Evalution:")
    print("Total loss: {}".format(total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision: %.4f" % (100 * correct_nums / total_nums) + '%')


def train_one_epoch_signal(model, epoch, epochs, device, train_loader, loss_func, optimizer, preprocess, scheduler):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train')
    for _, (window_datas, window_labels) in enumerate(loop): # shape(4,400,8)
        window_datas = window_datas.transpose(1, 2).unsqueeze(-1).to(device).type(torch.float32) # shape(4,8,400,1)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
        
        predicts = model(window_datas, text)
        
        labels = window_labels.to(device).type(torch.long) - 1
        loss = loss_func(predicts, labels)
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = loss.item())
    scheduler.step()
    print("[%d/%d] epoch's total loss = %f" % (epoch + 1, epochs, total_loss))
    save_results('res/results.csv', '%d, %12.6f\n' % (epoch + 1, total_loss))


def evaluate_signal(model, device, eval_loader, loss_func):
    model.eval() # 精度在64%
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Evaluating...")
    loop = tqdm(eval_loader, desc='Evaluation')
    for i, (window_data, window_labels) in enumerate(loop): # shape(16,400,8)
        window_data = window_data.transpose(1, 2).unsqueeze(-1).to(device).type(torch.float32) # shape(16,8,400,1)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
        
        predicts = model(window_data, text)
        
        labels = window_labels.to(device).type(torch.long) - 1
        loss = loss_func(predicts, labels)
        total_loss += loss

        predict_idx = predicts.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Evaluating [{i + 1}/{len(eval_loader)}]')
        loop.set_postfix(loss = loss.item())
    
    precision = '%.4f' % (100 * correct_nums / total_nums) + '%'
    print("Evalution:")
    print("Total loss: {}".format(total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision:", precision)
    return precision


def validate_signal(model, device, val_loader, loss_func):
    model.eval() # 精度在64%
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Validating...")
    loop = tqdm(val_loader, desc='Validation', ncols=100)
    for i, (window_data, window_labels) in enumerate(loop): # shape(16,400,8)
        window_data = window_data.transpose(1, 2).unsqueeze(-1).to(device).type(torch.float32) # shape(16,8,400,1)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
        
        predicts = model(window_data, text)
        
        labels = window_labels.to(device).type(torch.long) - 1
        loss = loss_func(predicts, labels)
        total_loss += loss

        predict_idx = predicts.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Validating [{i + 1}/{len(val_loader)}]')
        loop.set_postfix(loss = loss.item())
    
    precision = '%.4f' % (100 * correct_nums / total_nums) + '%'
    print("Validation:")
    print("Total loss: {}".format(total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision:", precision)
    return correct_nums.item() / total_nums



# train signal and text jointly
def train_one_epoch_signal_text(model, epoch, epochs, device, train_loader, loss_func, optimizer, preprocess, scheduler):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc='Train')
    for _, (window_datas, window_labels) in enumerate(loop): # shape(4,400,8)
        window_datas = window_datas.transpose(1, 2).unsqueeze(-1).to(device).type(torch.float32) # shape(4,8,400,1)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
        
        logits_per_image, logits_per_text = model(window_datas, text)
        
        labels = torch.LongTensor(range(len(window_labels))).to(device)
        loss_I = loss_func(logits_per_image, labels)
        loss_T = loss_func(logits_per_text, labels)
        loss = (loss_I + loss_T) / 2
        total_loss += loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
        loop.set_postfix(loss = loss.item())
    scheduler.step()
    print("[%d/%d] epoch's total loss = %f" % (epoch + 1, epochs, total_loss))
    save_results('res/results.csv', '%d, %12.6f\n' % (epoch + 1, total_loss))


def validate_signal_text(model, device, val_loader, loss_func):
    model.eval()
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Validating...")
    loop = tqdm(val_loader, desc='Train')
    for i, (window_data, window_labels) in enumerate(loop): # shape(4,400,8)
        window_data = window_data.transpose(1, 2).unsqueeze(-1).to(device).type(torch.float32) # shape(4,8,400,1)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
                
        logits_per_image, logits_per_text = model(window_data, text)
        
        labels = torch.LongTensor(range(len(window_labels))).to(device)
        loss_I = loss_func(logits_per_image, labels)
        loss_T = loss_func(logits_per_text, labels)
        loss = (loss_I + loss_T) / 2
        total_loss += loss

        predict_idx = logits_per_image.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Validating [{i + 1}/{len(val_loader)}]')
        loop.set_postfix(loss = loss.item())

    precision = '%.4f' % (100 * correct_nums / total_nums) + '%'
    print("Validation:")
    print("Total loss: {}".format(total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision:", precision)
    return correct_nums.item() / total_nums


def evaluate_signal_text(model, device, eval_loader, loss_func):
    model.eval() # 精度在64%
    total_loss, correct_nums, total_nums = 0.0, 0, 0
    
    print("Evaluating...")
    loop = tqdm(eval_loader, desc='Evaluation')
    for i, (window_data, window_labels) in enumerate(loop): # shape(16,400,8)
        text = clip.tokenize([template + prompts[mov_idx + 12] for mov_idx in window_labels]).to(device)
                
        logits_per_image, logits_per_text = model(window_data, text)
        
        labels = torch.LongTensor(range(len(window_labels))).to(device)
        loss_I = loss_func(logits_per_image, labels)
        loss_T = loss_func(logits_per_text, labels)
        loss = (loss_I + loss_T) / 2
        total_loss += loss

        predict_idx = logits_per_image.argmax(dim=-1)
        correct_nums += torch.sum(predict_idx == labels)
        total_nums += len(predict_idx)

        loop.set_description(f'Evaluating [{i + 1}/{len(eval_loader)}]')
        loop.set_postfix(loss = loss.item())
    
    precision = '%.4f' % (100 * correct_nums / total_nums) + '%'
    print("Evalution:")
    print("Total loss: {}".format(total_loss))
    print("Correct/Total: {}/{}".format(correct_nums, total_nums))
    print("Precision:", precision)
    return precision
