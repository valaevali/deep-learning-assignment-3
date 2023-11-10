import torch
import torch.nn as nn
import statistics
import torchvision.models as models
import torchtext.vocab as vocab
import random


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # use pretrained weights from resnet model
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size) # added batch normalization
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        features = self.resnet50(images)
        features = self.bn(features)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.GRU(embed_size, hidden_size, num_layers) # using the GRU architecture instead of the RNN
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        '''
        the input feature from the image is fed as the first input to RNN
        we apply teacher forcing ratio = 100% during training i.e. we do not feed the prediction of the previous step as a input to the current step
        '''
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        output = self.linear(hiddens)
        return output


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0) 
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
