"""condensate information

   - LF98 Lodders and Fegley (1998)

"""

#g/cm3
conddensity= { \
               "MgSiO":3.192, #LF98\
               "Fe_solid":7.875,#LF98\
}

if __name__ == "__main__":
    print(conddensity["Fe_solid"])
