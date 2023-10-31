# write a function to print 'Hello World!'
def custom_print_function(n):
    string = 'Hello World!'
    for i in range(n):
        print(string)
    
if __name__ == '__main__':
    n = int(input('How many times do you want to print Hello World?'))
    custom_print_function(n)