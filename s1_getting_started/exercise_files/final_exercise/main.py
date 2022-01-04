import argparse
import sys

import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.001)
        parser.add_argument('--epochs', default=10)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        critirion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_set, _ = mnist()
        losses = []
        for e in range(args.epochs):
            running_loss = 0
            for images, labels in train_set:
                out = model(images)
                loss = critirion(out, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
            losses.append(running_loss)
        plt.plot(losses, label="Loss")
        plt.legend()
        plt.title("Loss")
        plt.savefig("loss.png")
        torch.save(model.state_dict(), "trained_model.pt")


        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model_states = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(model_states)
        _, test_set = mnist()
        running_acc = []
        with torch.no_grad():
            for image, label in test_set:
                out = model(image)
                ps = out.exp()
                _, top_class = ps.topk(1, dim=1)
                equals = top_class == label.view(*top_class.shape)
                running_acc.append(equals)
        accuracy = torch.mean(torch.cat(running_acc).type(torch.FloatTensor))
        print(f'Accuracy: {accuracy.item()*100}%')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    