import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
# Create a graph
G = nx.Graph()

# Add nodes
G.add_node(1)
G.add_node(2)
G.add_node(3)

# Add edges
G.add_edge(1, 2)
G.add_edge(2, 3)

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()

A = nx.adjacency_matrix(G).todense()  # Adjacency matrix
X = np.identity(G.number_of_nodes())  # Node feature matrix

# Convert to PyTorch tensors.
A_tensor = torch.FloatTensor(A)
X_tensor = torch.FloatTensor(X)

adjacencies = [torch.FloatTensor(graph[0]) for graph in graphs]
features = [torch.FloatTensor(graph[1]) for graph in graphs]

model = MultiHeadGATLayer(nhead=8, in_features=10, out_features=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model.
for epoch in range(100):  # Number of epochs
    for i, (adjacency, feature) in enumerate(zip(adjacencies, features)):
        # Forward pass
        outputs = model(feature, adjacency)
        loss = criterion(outputs, labels[i])
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))