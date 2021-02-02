#Linear regresion script with only numpy
import numpy as np

#import matplotlib to make the graphic
import matplotlib.pyplot as plt


#Estimate the values of the linear function
def estimate_b_0_b_1(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    
    b_1 = np.sum((x - m_x)*(y - m_y)) / np.sum(x*(x - m_x))
    b_0 = m_y - b_1 * m_x
    
    return(b_0, b_1)


#graph the regresion
def plot_regresion(x, y, b):
    plt.scatter(x, y, color = 'g', marker = 'o', s = 30)
    y_pred = b[0] + b[1]*x
    print(y_pred)
    plt.plot(x, y_pred, color = 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.show()


#Call the regresion    
def main():
    x = np.array([1, 2, 3 ,4 ,5 ,6, 7, 8, 9, 10])
    y = np.array([1, 3, 5, 8, 11, 14, 13, 14, 19, 20])
    
    b = estimate_b_0_b_1(x, y)
    plot_regresion(x, y, b)
    
    
if __name__=="__main__":
    main()
