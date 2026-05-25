# Assignment - 4 (Part A)
# Determing rate and amount of tax to charge on the SUB-TOTAL
# Tax = GST(5%) + PST
# Export tax for international orders
# Create function CalculateOrderTax
# return total tax amount

class CalculateTax:
   
    def __init__(self, sub_total, country_code, province_code=""):
        '''
        Constructor to initialize the input variables
        '''
        self.sub_total = sub_total
        self.country_code = country_code.upper()
        self.province_code = province_code.upper()
        
    def CalculateOrderTax(self, sub_total, country_code, province_code=""):
        '''
        Function to calculate the total tax based on the country code, province code 
        (if any) and the sub total amount.
        '''
        # Creating a dictionary to store all the province code and their tax rates
        self.province_tax = {'AB': 0, 'BC': 0.07,
                             'ON': 0.08, 'MB': 0,
                             'NB': 0, 'NL': 0,
                             'PE': 0, 'YT': 0,
                             'QC': 0, 'NS': 0,
                             'NU': 0, 'NT': 0
                             }
        
        # Initializing the variables. GST is always 5% and export tax always 2%
        self.total_tax, self.export_tax, self.gst = 0, 0.02, 0.05
        
        # Checking if the order is domestic or International
        if self.country_code=='CA':
            # Province mb has different tax rates based on the sub total
            # Therefore, updating the province tax of MB is sub total > $20
            if self.province_code=='MB':
                if self.sub_total>20:
                    self.province_tax['MB'] = 0.07
                    
            # Calculating the total tax for domestic orders
            self.total_tax = self.sub_total*(self.province_tax[province_code] + self.gst)
        else:
            # Calculating the total tax for International orders
            self.total_tax = self.sub_total*self.export_tax
            
        return self.total_tax
    
    def display(self):
        '''
        Function to display the result in the given format
        '''
        print(f"Shipping To:      \t{self.country_code} {self.province_code}")
        
        if self.country_code=='CA':
            print("Destination Type: \tDomestic")
        else:
            print("Destination Type: \tInternational")
            
        print(f"Order sub total: \t${self.sub_total}")
        total_tax = self.CalculateOrderTax(self.sub_total, self.country_code, self.province_code)
        print(f"Total Tax Charged: \t${total_tax}")
        print(f"Order Grand Total: \t${self.sub_total+total_tax}")
            
        

sub_total = float(input("Enter the sub total amount: "))
country_code = input("Enter the country code: ")
province_code = " "
if country_code.upper()=='CA':
    province_code = input("Enter the province code: ")
test = CalculateTax(sub_total, country_code, province_code)
test.display()


###############################################################################

# Enter the sub total amount: 1500

# Enter the country code: US
# Shipping To:      	US  
# Destination Type: 	International
# Order sub total: 	$1500.0
# Total Tax Charged: 	$30.0
# Order Grand Total: 	$1530.0


# Enter the sub total amount: 1100

# Enter the country code: ca

# Enter the province code: on
# Shipping To:      	CA ON
# Destination Type: 	Domestic
# Order sub total: 	$1100.0
# Total Tax Charged: 	$143.0
# Order Grand Total: 	$1243.0
