# khj-ai

net = torchvision.models.resnet18(pretrained = True)
resnet18로 모델을 설정해준다.





net.fc = nn.Linear(net.fc.in_features,3)
net = net.to(0)
resnet18은 마지막 linear단계에서 out_feature의 수가 1000이므로 3으로 수정해준다.





criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.0001)
#optimizer = optim.SGD(net.parameters(), lr=0.001)
Adam optimizer가 너 좋은 성능을 보이므로 Adam사용 






 for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(0), labels.to(0)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        temp += (predicted == labels).sum().item()
        total += inputs.shape[0]
train dateset을 통해서 모델을 훈련시키는 부분







correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(0), labels.to(0)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('test셋 결과물: %d %%' % (
    100 * correct / total))

test dataset을 통해서 정확도를 검사하는 부분





mat = torch.zeros((3,3))
class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(0), labels.to(0)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted, labels)
        c = (predicted == labels).squeeze()
        for i in range(2):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
        for i,x in enumerate(labels):
          mat[predicted[i],x] += 1
print(mat)
for i in range(3):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
        
normal, covid, viral pneumonia 별로 정확도를 측정하고 tesor형식으로 이미지들을 어떤 label로 판별했는지 보여주는 




