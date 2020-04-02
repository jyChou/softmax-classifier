import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=" ")
    return data[:, 1:], data[:, 0].astype(int)

def plot_loss_and_accuracy(clr):
    def plot_cross_entropy_loss(clr, ax):
        training_loss = ax.plot( clr.loss['train'], label='training loss')
        testing_loss = ax.plot(clr.loss['test'], label='testing loss')
        ax.legend()
        ax.set_title('Cross Entory Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
                     
        return ax
    
    def plot_mpc_accuracy(clr, ax):
        ax.plot(clr.acc['train'], label='training accuracy')
        ax.plot(clr.acc['test'], label='testing accuracy')
        ax.legend()
        ax.set_title('Mean Per-Class Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        return ax
    
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0] = plot_cross_entropy_loss(clr, ax[0])
    ax[1] = plot_mpc_accuracy(clr, ax[1])
    plt.savefig("2-1")
    

def plot_boundary(clr, data_x, data_y, n=512):
    
    def f(x, y, clr):
        x = np.tile(x, n)
        y = np.repeat(y, n)
        tmp = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)
        z = clr.predict(tmp).reshape(n,n)
        return z
    
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    fig, ax = plt.subplots()
    # draw contour lines and fill contours
    cs = ax.contourf( x, y, f(x, y, clr) ,colors=['green'], alpha=0.35, extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    # create scatter plot 
    color = np.array(['blue', 'green', 'red'])
    ax.scatter(data_x[:,0], data_x[:, 1], c=color[data_y - 1], s=7)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title('Decision boundaries (training data)')
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    plt.tight_layout()
    plt.savefig("2-2")