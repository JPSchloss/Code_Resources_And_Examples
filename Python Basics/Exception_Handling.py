def main():
    try:
        # Test examples that will raise particular errors. 
        # The code will halt at the first error. 
        # So be sure to test each individually. 
        x = "1" / 0  # TypeError
        y = [1, 2, 3][5]  # IndexError
        z = {"key": "value"}["no_key"]  #KeyError

    except TypeError:
        # Handle the TypeError exception
        print("Invalid operation due to inappropriate type!")
        
    except IndexError:
        # Handle the IndexError exception
        print("Tried to access an out-of-range index!")
        
    except KeyError:
        # Handle the KeyError exception
        print("Tried to access a dictionary with a non-existing key!")
        
    except:
        # Handle all other exceptions
        print("Something went wrong!")

    return

main()