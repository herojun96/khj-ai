# khj-ai

net = torchvision.models.resnet18(pretrained = True)
resnet18로 모델을 설정해준다.

net.fc = nn.Linear(net.fc.in_features,3)
net = net.to(0)
resnet18은 마지막 linear단계에서 out_feature의 수가 1000이므로 3으로 수정해준다.

