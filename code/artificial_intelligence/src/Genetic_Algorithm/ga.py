from csv import reader
from ga_helper import GA



def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
	
def load_csv(filename):
    
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
	
    for j in range(len(dataset[0])):
        str_column_to_float(dataset,j)


    return dataset
                

sample = load_csv('iris.csv')
ga = GA(data=sample,
              countclusters=3,
              chromosomecount=100,
              populationcount=150,
              countmutationgens=1
            )

ga.run()
accuracy=0
x,y,z=0,0,0
clusters = ga.getclusters()
for i in range(len(clusters)):
    print('Cluster %d' %(i+1))
    x,y,z=0,0,0
    print(clusters[i])
    for j in clusters[i]:
        if j < 50:
            x+=1
        elif j<100:
            y+=1
        elif j<150:
            z+=1
    accuracy+=max(x,y,z)
accuracy = accuracy / len(sample)
print('Accuracy is : ',end='')
print(accuracy*100,'%')
