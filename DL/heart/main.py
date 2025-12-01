

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

print(data)
print(x_data)

x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

t = torch.ones(4)
print(t)

x = t.numpy()

print(x)