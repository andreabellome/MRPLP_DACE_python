import pandas as pd
import numpy as np
import os

class SatelliteEphemeris:
    def __init__(self, filename=None):
        """
        Initialize the SatelliteEphemeris class.
        If filename is provided, the ephemeris data will be loaded from the file.
        Otherwise, it loads a default debris file.
        """
        self.filename = filename
        self.debris = None

    @staticmethod
    def jd_to_mjd2000(jd):
        """
        Convert Julian Date (JD) to Modified Julian Date (MJD2000).
        MJD2000 = JD - 2451544.5
        """
        return jd - 2400000.5 - 51544.5

    @staticmethod
    def juliandate(date):
        """
        Convert a date to Julian Date (JD).
        The input date should be a datetime object or array of datetime objects.
        """
        jd = date.to_julian_date() if isinstance(date, pd.Timestamp) else date.apply(pd.Timestamp.to_julian_date)
        return jd

    @staticmethod
    def deg2rad(degrees):
        """
        Convert degrees to radians.
        """
        return np.deg2rad(degrees)

    @staticmethod
    def M2theta(M, e):
        """
        Convert Mean Anomaly (M) to True Anomaly (theta).
        """
        E = M
        it = 0
        itMax = 15000
        
        if e < 1:  # Elliptical orbit
            while abs((M - E + e * np.sin(E)) > 1e-13) and it < itMax:
                ddf = (e * np.cos(E) - 1)
                E -= (M - E + e * np.sin(E)) / ddf
                it += 1
            theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        else:  # Hyperbolic orbit
            for _ in range(20):
                ddf = (1 - e * np.cosh(E))
                E -= (M + E - e * np.sinh(E)) / ddf
                theta = 2 * np.arctan(np.sqrt((1 + e) / (e - 1)) * np.tanh(E / 2))

        return theta

    def hms_to_frac_day(self, hours, minutes, seconds):
        """
        Convert hours, minutes, and seconds to a fractional day.
        """
        return (hours + (minutes + seconds/60)/60) / 24
    
    def date_to_jd(self, date):
        """
        Convert a Gregorian date to Julian Date (JD).
        The input date should be a list or array-like of [year, month, day, hour, minute, second].
        """
        Y, M, D, hrs, mn, sec = date

        # Validate input
        if Y < -4713 or (Y == -4713 and (M < 11 or (M == 11 and (D < 24 or (D == 24 and hrs < 12))))):
            raise ValueError("The function is valid for dates after 12:00 noon, 24 November -4713.")

        # Convert Gregorian date to Julian Date using the formula
        jd = (367 * Y - np.floor(7 * (Y + np.floor((M + 9) / 12)) / 4) -
            np.floor(3 * np.floor((Y + (M - 9) / 7) / 100 + 1) / 4) +
            np.floor(275 * M / 9) +
            D + 1721028.5 + self.hms_to_frac_day(hrs, mn, sec))
        
        return jd
    
    def timestamp_to_mjd2000(self, timestamp):
        """
        Convert a pandas Timestamp to Modified Julian Date (MJD2000).
        """
        # Extract date components
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        minute = timestamp.minute
        second = timestamp.second

        # Convert to Julian Date
        jd = self.date_to_jd([year, month, day, hour, minute, second])
        
        # Calculate MJD2000
        mjd2000 = jd - 2400000.5 - 51544.5  # MJD2000 = JD - 2451544.5
        
        return mjd2000

    def clean_data(self, data):
        """
        Clean the data by keeping the first column as string (satellite IDs),
        and converting the remaining columns to float.
        """
        cleaned_data = []
        for line in data:
            cleaned_line = []
            for i, item in enumerate(line):
                cleaned_item = item.replace(',', '').strip()
                if i == 0:
                    # First column (satellite IDs) remains as a string
                    cleaned_line.append(cleaned_item)
                else:
                    try:
                        cleaned_line.append(float(cleaned_item))  # Convert to float for other columns
                    except ValueError:
                        print(f"Error: Could not convert '{item}' to float. Skipping this item.")
                        continue
            cleaned_data.append(cleaned_line)
        return cleaned_data

    def load_data(self):
        """
        Load the satellite data from the provided Excel file or a default debris file.
        """
        try:
            if self.filename:
                # Read satellite data from Excel file
                sats = pd.read_excel(self.filename)
                idsats = sats.iloc[:, 0].astype(str)  # Keep satellite IDs as strings

                coes = sats.iloc[:, 4:10].to_numpy()  # Orbital elements (columns 5 to 10)
                coes[:, 2:6] = self.deg2rad(coes[:, 2:6])  # Convert degrees to radians (columns 3 to 6)

                # Convert dates to Julian Date and then to MJD2000
                dates = pd.to_datetime(sats.iloc[:, 2])  # Column 3 contains dates

                epoch_list = []
                for date in dates:
                    if not isinstance(date, pd.Timestamp):
                        date = pd.Timestamp(date)
                        epoch_list.append(self.timestamp_to_mjd2000(date))
                    else:
                        epoch_list.append(self.timestamp_to_mjd2000(date))
                epochs = np.array(epoch_list)

                # Create debris matrix with IDs, epochs, and orbital elements (coes)
                self.debris = np.column_stack((idsats, epochs, coes))

                for ind, deb in enumerate(self.debris):
                    self.debris[ind][0] = str(ind)
                    
                print('Custom database loaded.')
            else:
                # Load default debris data from the Debris.txt file in the ephemerides folder
                base_dir = os.path.dirname(__file__)  # Get the directory of the current script
                filepath = os.path.join(base_dir, 'ephemerides', 'Debris.txt')

                # Read and clean data
                with open(filepath, 'r') as f:
                    raw_data = [line.split() for line in f.readlines()]
                self.debris = self.clean_data(raw_data)
                self.debris = np.array( self.debris, dtype=object )
                for ind, deb in enumerate(self.debris):
                    self.debris[ind][0] = str(ind)
                print('GTOC9 database loaded.')
        except Exception as e:
            print(f"Error loading data: {e}")

    def get_debris(self):
        """
        Return the debris matrix. Call this after loading the data.
        """
        return self.debris
    
    
    def eph_satellites(self, tf, iD):
        """
        Calculate Cartesian coordinates of the satellites based on the debris data.
        :param tf: Time in seconds.
        :param iD: Index of the satellite.
        :return: np.array with Cartesian coordinates.
        """
        # Load constants
        mue = 398600.4418e9  # [m^3/s^2]
        Re = 6378137  # [m]
        J2 = 1.08262668e-3

        t0 = self.debris[int(iD), 1] * 86400  # [s]
        a = self.debris[int(iD), 2]  # [m]
        e = self.debris[int(iD), 3]  # Eccentricity
        inc = self.debris[int(iD), 4]  # [rad]
        Om0 = self.debris[int(iD), 5]  # [rad]
        om0 = self.debris[int(iD), 6]  # [rad]
        M0 = self.debris[int(iD), 7]  # [rad]

        n = np.sqrt(mue / (a ** 3))
        p = a * (1 - e ** 2)

        Omdot = -3 / 2 * J2 * n * ((Re / p) ** 2) * np.cos(inc)
        omdot = 3 / 4 * n * J2 * ((Re / p) ** 2) * (5 * (np.cos(inc) ** 2) - 1)

        Om = Om0 + Omdot * (tf - t0)
        om = om0 + omdot * (tf - t0)
        M = M0 + n * (tf - t0)

        # Normalize angles to be within [0, 2*pi]
        Om = Om % (2 * np.pi)
        om = om % (2 * np.pi)
        M = M % (2 * np.pi)

        # Convert Mean Anomaly to True Anomaly
        th = self.M2theta(M, e)
        gam = np.arctan(e * np.sin(th) / (1 + e * np.cos(th)))

        r = p / (1 + e * np.cos(th))
        v = np.sqrt(2 * mue / r - mue / a)

        x = r * (np.cos(th + om) * np.cos(Om) - np.sin(th + om) * np.cos(inc) * np.sin(Om))
        y = r * (np.cos(th + om) * np.sin(Om) + np.sin(th + om) * np.cos(inc) * np.cos(Om))
        z = r * (np.sin(th + om) * np.sin(inc))

        vx = v * (-np.sin(th + om - gam) * np.cos(Om) - np.cos(th + om - gam) * np.cos(inc) * np.sin(Om))
        vy = v * (-np.sin(th + om - gam) * np.sin(Om) + np.cos(th + om - gam) * np.cos(inc) * np.cos(Om))
        vz = v * (np.cos(th + om - gam) * np.sin(inc))

        return np.array([x, y, z, vx, vy, vz])
    
