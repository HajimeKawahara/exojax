from radis.api.dbmanager import DatabaseManager
import requests   
from io import StringIO
import pandas as pd


class CustomDatabaseManager(DatabaseManager):  
    """common base class for custom database managers"""
    def __init__(  
        self,  
        name,  
        molecule,  
        local_databases,
        url,  
        engine="default",  
        verbose=True,  
        parallel=True,  
    ):  
        """initialize the custom database manager
        Args:
            name (str): Name of the database
            molecule (str): Molecule name
            local_databases (str): Path to the local databases
            url (str): URL of the data file
            engine (str): Computational engine to use
            verbose (bool): Verbosity flag
            parallel (bool): Parallel processing flag
        """
        super().__init__(  
            name,  
            molecule,  
            local_databases,  
            engine,  
            verbose=verbose,  
            parallel=parallel,  
        )  
        self.downloadable = True  
        self.url = url   
      
    def fetch_urlnames(self):  
        """fetch the URL names of the data files

        Returns:
            list: List of URL names
        """
        return [self.url]  
    
    def download(self, urlname):
        """download the data file from the given URL
        
        Args:
            urlname (str): URL of the data file
        Returns:
            response (requests.Response): Response object containing the downloaded data
        """
        response = requests.get(urlname)  
        response.raise_for_status() 
        return response

      
class HargreavesDatabaseManager(CustomDatabaseManager):
    """database manager for the line list of the transition of FeH (Hargreaves et al., 2010)"""
    def __init__(
        self,  
        name,  
        molecule,  
        local_databases,
        url,  
        engine="default",  
        verbose=True,  
        parallel=True,  
    ):  
        """initialize the Hargreaves database manager

        Args:
            name (str): Name of the database
            molecule (str): Molecule name
            local_databases (str): Path to the local databases
            url (str): URL of the data file
            engine (str): Computational engine to use
            verbose (bool): Verbosity flag
            parallel (bool): Parallel processing flag
        """
        super().__init__(  
            name,  
            molecule,  
            local_databases,
            url,  
            engine,  
            verbose,  
            parallel,   
        )     
    
    def parse_to_local_file(  
        self,   
        urlname,  
        local_file,   
    ):  
        """download and parse the data file to a local file

        Args:
            urlname (str): URL of the data file
            local_file (str): Local file path to save the data

        Returns:
            int: Number of lines in the downloaded data

        """  
        writer = self.get_datafile_manager()  
          
        # Download the data file  
        response = self.download(urlname)
 
        # Read the data into a DataFrame  
        columns = ["wavenumber", "intensity", "e_lower", "einsteinA", "j_lower", "branch", "omega"]  
        df = pd.read_csv(StringIO(response.text), sep='\s+', skiprows=26 ,header=None, names=columns)
 
        wmin = df["wavenumber"].min()
        wmax = df["wavenumber"].max()
        Nlines = len(df) 
          
        # write the data to a local file  
        writer.write(local_file, df, append=False)  
          
        # combine the temporary batch files 
        writer.combine_temp_batch_files(local_file)  
          
        # add metadata to the local file  
        from radis import __version__  
          
        writer.add_metadata(  
            local_file,  
            {  
                "wavenumber_min": wmin,
                "wavenumber_max": wmax,
                "download_date": self.get_today(),  
                "download_url": urlname,  
                "version": __version__,  
            },  
        )  
            
        return Nlines  
    