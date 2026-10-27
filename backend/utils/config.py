"""
Research-grade configuration for Indian Stock Recommendation System
══════════════════════════════════════════════════════════════════════
Expanded universe: NIFTY 50 + NIFTY Next 50 + NIFTY Midcap 100 +
popular Small/Micro-cap + Sector-specific NSE-listed stocks.
"""
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class StockUniverse:
    # ── NIFTY 50 ──────────────────────────────────────────────────────────────
    NIFTY_50: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
        "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS", "HCLTECH.NS",
        "WIPRO.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "TITAN.NS", "POWERGRID.NS",
        "NTPC.NS", "ONGC.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "COALINDIA.NS",
        "GRASIM.NS", "ADANIENT.NS", "ADANIPORTS.NS", "HINDALCO.NS", "CIPLA.NS",
        "DRREDDY.NS", "DIVISLAB.NS", "EICHERMOT.NS", "BPCL.NS", "BRITANNIA.NS",
        "HEROMOTOCO.NS", "HINDUNILVR.NS", "BAJAJFINSV.NS", "BAJAJ-AUTO.NS", "APOLLOHOSP.NS",
        "NESTLEIND.NS", "SBILIFE.NS", "HDFCLIFE.NS", "TECHM.NS", "INDUSINDBK.NS",
        "UPL.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "LTIM.NS", "SHRIRAMFIN.NS",
    ])

    # ── NIFTY Next 50 ─────────────────────────────────────────────────────────
    NIFTY_NEXT_50: List[str] = field(default_factory=lambda: [
        "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
        "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS", "BOSCHLTD.NS",
        "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS",
        "DLF.NS", "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS",
        "ICICIPRULI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "INDHOTEL.NS",
        "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS",
        "JSWENERGY.NS", "JUBLFOOD.NS", "LICI.NS", "LUPIN.NS",
        "M&M.NS", "MARICO.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS",
        "NAUKRI.NS", "NHPC.NS", "NMDC.NS", "OBEROIRLTY.NS",
        "OFSS.NS", "PAGEIND.NS", "PEL.NS", "PERSISTENT.NS",
        "PIDILITIND.NS", "PNB.NS", "POLYCAB.NS", "RECLTD.NS",
        "SAIL.NS", "SBICARD.NS", "SIEMENS.NS", "TORNTPHARM.NS",
        "TRENT.NS", "VEDL.NS",
    ])

    # ── NIFTY Midcap 100 (select) ─────────────────────────────────────────────
    NIFTY_MIDCAP: List[str] = field(default_factory=lambda: [
        "AARTIIND.NS", "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS",
        "ACC.NS", "ALKEM.NS", "ASHOKLEY.NS", "ASTRAL.NS",
        "ATUL.NS", "AUBANK.NS", "BALKRISIND.NS", "BATAINDIA.NS",
        "BHARATFORG.NS", "BIOCON.NS", "CANFINHOME.NS", "CENTRALBK.NS",
        "CGPOWER.NS", "CHAMBLFERT.NS", "CONCOR.NS", "COROMANDEL.NS",
        "CROMPTON.NS", "CUB.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS",
        "DELHIVERY.NS", "DEVYANI.NS", "DIXON.NS", "ESCORTS.NS",
        "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", "GLENMARK.NS",
        "GMRINFRA.NS", "GNFC.NS", "GODREJPROP.NS", "GRANULES.NS",
        "GSPL.NS", "HAL.NS", "HFCL.NS", "HINDPETRO.NS",
        "HONAUT.NS", "IBREALEST.NS", "IDBI.NS", "IEX.NS",
        "INDIAMART.NS", "INDIANB.NS", "IRFC.NS", "IPCALAB.NS",
        "JKCEMENT.NS", "KALYANKJIL.NS", "KEI.NS", "KPITTECH.NS",
        "L&TFH.NS", "LAURUSLABS.NS", "LICHSGFIN.NS", "LTTS.NS",
        "MANAPPURAM.NS", "MFSL.NS", "MGL.NS", "MPHASIS.NS",
        "MRF.NS", "NAM-INDIA.NS", "NATCOPHARM.NS", "NATIONALUM.NS",
        "NAVINFLUOR.NS", "NIACL.NS", "NLCINDIA.NS", "NOCIL.NS",
        "NYKAA.NS", "PAYTM.NS", "PETRONET.NS", "PFC.NS",
        "PIIND.NS", "PRESTIGE.NS", "PVRINOX.NS", "RAJESHEXPO.NS",
        "RAMCOCEM.NS", "RBLBANK.NS", "SONACOMS.NS", "SOLARINDS.NS",
        "SRF.NS", "SUPREMEIND.NS", "SUNTV.NS", "SYNGENE.NS",
        "TATACHEM.NS", "TATACOMM.NS", "TATAELXSI.NS", "TATAPOWER.NS",
        "TORNTPOWER.NS", "TVSMOTOR.NS", "UBL.NS", "UNIONBANK.NS",
        "UNITDSPR.NS", "VOLTAS.NS", "WHIRLPOOL.NS", "ZEEL.NS",
        "ZOMATO.NS", "ZYDUSLIFE.NS",
    ])

    # ── Popular Small & Micro-Cap / New-Age / Sector Picks ────────────────────
    POPULAR_STOCKS: List[str] = field(default_factory=lambda: [
        # New-age / Tech / Fintech
        "DMART.NS", "NYKAA.NS", "POLICYBZR.NS", "CARTRADE.NS",
        "EASEMYTRIP.NS", "RATEGAIN.NS", "HAPPSTMNDS.NS", "ROUTE.NS",
        "NAZARA.NS", "KAYNES.NS", "AFFLE.NS", "TANLA.NS",
        "LATENTVIEW.NS", "MAPMYINDIA.NS",

        # Auto & EV
        "MOTHERSON.NS", "MRF.NS", "BALKRISIND.NS",
        "OLECTRA.NS", "TATAELXSI.NS", "TATAMOTORS.NS",

        # Pharma / Healthcare
        "MAXHEALTH.NS", "METROPOLIS.NS", "LALPATHLAB.NS",
        "SYNGENE.NS", "GLAND.NS", "MEDANTA.NS",

        # FMCG / Consumer
        "TATACONSUM.NS", "VBL.NS", "RADICO.NS", "GODREJCP.NS",
        "EMAMILTD.NS", "JYOTHYLAB.NS",

        # Real Estate / Infra
        "LODHA.NS", "PHOENIXLTD.NS", "BRIGADE.NS", "SOBHA.NS",
        "IRB.NS", "PNC.NS", "COCHINSHIP.NS",

        # Defence / PSU
        "HAL.NS", "BEL.NS", "MAZAGON.NS", "BDL.NS",
        "GRSE.NS", "IRFC.NS", "RVNL.NS", "RAILTEL.NS",
        "RITES.NS", "NHPC.NS", "SJVN.NS",

        # Chemicals / Specialty
        "CLEAN.NS", "AARTI.NS", "DEEPAKNTR.NS",
        "TATACHEM.NS", "PIIND.NS", "SRF.NS",

        # Energy / Renewable
        "ADANIGREEN.NS", "TATAPOWER.NS", "NHPC.NS",
        "JSWENERGY.NS", "CESC.NS", "TORNTPOWER.NS",

        # Financials / NBFCs
        "BAJFINANCE.NS", "CHOLAFIN.NS", "M&MFIN.NS", "POONAWALLA.NS",
        "IIFL.NS", "ANGELONE.NS", "CDSL.NS", "BSE.NS",
        "MUTHOOTFIN.NS", "MANAPPURAM.NS",

        # Metals & Mining
        "VEDL.NS", "HINDZINC.NS", "NATIONALUM.NS", "NMDC.NS",
        "MOIL.NS",

        # Telecom / Media
        "IDEA.NS", "TTML.NS", "ZEEL.NS", "SUNTV.NS",

        # Cement / Building Materials
        "AMBUJACEM.NS", "ACC.NS", "JKCEMENT.NS",
        "RAMCOCEM.NS", "DALBHARAT.NS", "JKLAKSHMI.NS",

        # Textiles / Retail
        "TRENT.NS", "ABFRL.NS", "PAGEIND.NS", "SHOPERSTOP.NS",
        "VMART.NS",

        # Agri / Fertiliser
        "CHAMBLFERT.NS", "COROMANDEL.NS", "UPL.NS",
        "PIIND.NS", "GNFC.NS",

        # Insurance
        "SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS",
        "STARHEALTH.NS", "NIACL.NS",

        # Logistics / Transport
        "DELHIVERY.NS", "CONCOR.NS", "BLUEDART.NS",
        "GESHIP.NS", "ALLCARGO.NS",

        # Additional popular stocks
        "SUZLON.NS", "YESBANK.NS", "BIOCON.NS",
        "BANDHANBNK.NS", "INDIGO.NS", "SPARC.NS",
        "IRCTC.NS", "IEX.NS",
    ])

    # ── Full Company Names (for display and search) ───────────────────────────
    COMPANY_NAMES: Dict[str, str] = field(default_factory=lambda: {
        # NIFTY 50
        "RELIANCE.NS": "Reliance Industries Ltd", "TCS.NS": "Tata Consultancy Services Ltd",
        "HDFCBANK.NS": "HDFC Bank Ltd", "INFY.NS": "Infosys Ltd",
        "ICICIBANK.NS": "ICICI Bank Ltd", "SBIN.NS": "State Bank of India",
        "BHARTIARTL.NS": "Bharti Airtel Ltd", "ITC.NS": "ITC Ltd",
        "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd", "LT.NS": "Larsen & Toubro Ltd",
        "AXISBANK.NS": "Axis Bank Ltd", "ASIANPAINT.NS": "Asian Paints Ltd",
        "MARUTI.NS": "Maruti Suzuki India Ltd", "BAJFINANCE.NS": "Bajaj Finance Ltd",
        "HCLTECH.NS": "HCL Technologies Ltd", "WIPRO.NS": "Wipro Ltd",
        "SUNPHARMA.NS": "Sun Pharmaceutical Industries Ltd", "ULTRACEMCO.NS": "UltraTech Cement Ltd",
        "TITAN.NS": "Titan Company Ltd", "POWERGRID.NS": "Power Grid Corporation of India Ltd",
        "NTPC.NS": "NTPC Ltd", "ONGC.NS": "Oil & Natural Gas Corporation Ltd",
        "TATASTEEL.NS": "Tata Steel Ltd", "JSWSTEEL.NS": "JSW Steel Ltd",
        "COALINDIA.NS": "Coal India Ltd", "GRASIM.NS": "Grasim Industries Ltd",
        "ADANIENT.NS": "Adani Enterprises Ltd", "ADANIPORTS.NS": "Adani Ports & SEZ Ltd",
        "HINDALCO.NS": "Hindalco Industries Ltd", "CIPLA.NS": "Cipla Ltd",
        "DRREDDY.NS": "Dr. Reddy's Laboratories Ltd", "DIVISLAB.NS": "Divi's Laboratories Ltd",
        "EICHERMOT.NS": "Eicher Motors Ltd", "BPCL.NS": "Bharat Petroleum Corporation Ltd",
        "BRITANNIA.NS": "Britannia Industries Ltd", "HEROMOTOCO.NS": "Hero MotoCorp Ltd",
        "HINDUNILVR.NS": "Hindustan Unilever Ltd", "BAJAJFINSV.NS": "Bajaj Finserv Ltd",
        "BAJAJ-AUTO.NS": "Bajaj Auto Ltd", "APOLLOHOSP.NS": "Apollo Hospitals Enterprise Ltd",
        "NESTLEIND.NS": "Nestle India Ltd", "SBILIFE.NS": "SBI Life Insurance Co Ltd",
        "HDFCLIFE.NS": "HDFC Life Insurance Co Ltd", "TECHM.NS": "Tech Mahindra Ltd",
        "INDUSINDBK.NS": "IndusInd Bank Ltd", "UPL.NS": "UPL Ltd",
        "TATACONSUM.NS": "Tata Consumer Products Ltd", "TATAMOTORS.NS": "Tata Motors Ltd",
        "LTIM.NS": "LTIMindtree Ltd", "SHRIRAMFIN.NS": "Shriram Finance Ltd",

        # NIFTY Next 50
        "ADANIGREEN.NS": "Adani Green Energy Ltd", "ADANIPOWER.NS": "Adani Power Ltd",
        "AMBUJACEM.NS": "Ambuja Cements Ltd", "AUROPHARMA.NS": "Aurobindo Pharma Ltd",
        "BANKBARODA.NS": "Bank of Baroda", "BEL.NS": "Bharat Electronics Ltd",
        "BERGEPAINT.NS": "Berger Paints India Ltd", "BOSCHLTD.NS": "Bosch Ltd",
        "CANBK.NS": "Canara Bank", "CHOLAFIN.NS": "Cholamandalam Investment & Finance Co Ltd",
        "COLPAL.NS": "Colgate-Palmolive India Ltd", "DABUR.NS": "Dabur India Ltd",
        "DLF.NS": "DLF Ltd", "GAIL.NS": "GAIL India Ltd",
        "GODREJCP.NS": "Godrej Consumer Products Ltd", "HAVELLS.NS": "Havells India Ltd",
        "ICICIPRULI.NS": "ICICI Prudential Life Insurance Co Ltd", "IDEA.NS": "Vodafone Idea Ltd",
        "IDFCFIRSTB.NS": "IDFC First Bank Ltd", "INDHOTEL.NS": "Indian Hotels Co Ltd",
        "INDUSTOWER.NS": "Indus Towers Ltd", "IOC.NS": "Indian Oil Corporation Ltd",
        "IRCTC.NS": "Indian Railway Catering & Tourism Corp Ltd", "JINDALSTEL.NS": "Jindal Steel & Power Ltd",
        "JSWENERGY.NS": "JSW Energy Ltd", "JUBLFOOD.NS": "Jubilant FoodWorks Ltd",
        "LICI.NS": "Life Insurance Corporation of India", "LUPIN.NS": "Lupin Ltd",
        "M&M.NS": "Mahindra & Mahindra Ltd", "MARICO.NS": "Marico Ltd",
        "MOTHERSON.NS": "Motherson Sumi Wiring India Ltd", "MUTHOOTFIN.NS": "Muthoot Finance Ltd",
        "NAUKRI.NS": "Info Edge India Ltd (Naukri)", "NHPC.NS": "NHPC Ltd",
        "NMDC.NS": "NMDC Ltd", "OBEROIRLTY.NS": "Oberoi Realty Ltd",
        "OFSS.NS": "Oracle Financial Services Software Ltd", "PAGEIND.NS": "Page Industries Ltd",
        "PEL.NS": "Piramal Enterprises Ltd", "PERSISTENT.NS": "Persistent Systems Ltd",
        "PIDILITIND.NS": "Pidilite Industries Ltd", "PNB.NS": "Punjab National Bank",
        "POLYCAB.NS": "Polycab India Ltd", "RECLTD.NS": "REC Ltd",
        "SAIL.NS": "Steel Authority of India Ltd", "SBICARD.NS": "SBI Cards & Payment Services Ltd",
        "SIEMENS.NS": "Siemens Ltd", "TORNTPHARM.NS": "Torrent Pharmaceuticals Ltd",
        "TRENT.NS": "Trent Ltd", "VEDL.NS": "Vedanta Ltd",

        # Midcap & Popular
        "ABB.NS": "ABB India Ltd", "ACC.NS": "ACC Ltd",
        "ASHOKLEY.NS": "Ashok Leyland Ltd", "ASTRAL.NS": "Astral Ltd",
        "BATAINDIA.NS": "Bata India Ltd", "BHARATFORG.NS": "Bharat Forge Ltd",
        "BIOCON.NS": "Biocon Ltd", "CGPOWER.NS": "CG Power & Industrial Solutions Ltd",
        "CUMMINSIND.NS": "Cummins India Ltd", "DEEPAKNTR.NS": "Deepak Nitrite Ltd",
        "DELHIVERY.NS": "Delhivery Ltd", "DIXON.NS": "Dixon Technologies India Ltd",
        "ESCORTS.NS": "Escorts Kubota Ltd", "FEDERALBNK.NS": "Federal Bank Ltd",
        "FORTIS.NS": "Fortis Healthcare Ltd", "GLENMARK.NS": "Glenmark Pharmaceuticals Ltd",
        "GODREJPROP.NS": "Godrej Properties Ltd", "HAL.NS": "Hindustan Aeronautics Ltd",
        "HINDPETRO.NS": "Hindustan Petroleum Corporation Ltd", "HONAUT.NS": "Honeywell Automation India Ltd",
        "IEX.NS": "Indian Energy Exchange Ltd", "IRFC.NS": "Indian Railway Finance Corporation Ltd",
        "IPCALAB.NS": "IPCA Laboratories Ltd", "KPITTECH.NS": "KPIT Technologies Ltd",
        "LAURUSLABS.NS": "Laurus Labs Ltd", "LICHSGFIN.NS": "LIC Housing Finance Ltd",
        "LTTS.NS": "L&T Technology Services Ltd", "MANAPPURAM.NS": "Manappuram Finance Ltd",
        "MGL.NS": "Mahanagar Gas Ltd", "MPHASIS.NS": "Mphasis Ltd",
        "MRF.NS": "MRF Ltd", "NATIONALUM.NS": "National Aluminium Co Ltd",
        "NYKAA.NS": "FSN E-Commerce Ventures Ltd (Nykaa)", "PAYTM.NS": "One97 Communications Ltd (Paytm)",
        "PETRONET.NS": "Petronet LNG Ltd", "PFC.NS": "Power Finance Corporation Ltd",
        "PIIND.NS": "PI Industries Ltd", "PVRINOX.NS": "PVR INOX Ltd",
        "SRF.NS": "SRF Ltd", "TATAPOWER.NS": "Tata Power Company Ltd",
        "TATAELXSI.NS": "Tata Elxsi Ltd", "TATACHEM.NS": "Tata Chemicals Ltd",
        "TVSMOTOR.NS": "TVS Motor Company Ltd", "VOLTAS.NS": "Voltas Ltd",
        "ZEEL.NS": "Zee Entertainment Enterprises Ltd", "ZOMATO.NS": "Zomato Ltd",
        "ZYDUSLIFE.NS": "Zydus Lifesciences Ltd",

        # New-age / Defence / Popular
        "DMART.NS": "Avenue Supermarts Ltd (DMart)", "SUZLON.NS": "Suzlon Energy Ltd",
        "YESBANK.NS": "Yes Bank Ltd", "INDIGO.NS": "InterGlobe Aviation Ltd (IndiGo)",
        "ANGELONE.NS": "Angel One Ltd", "CDSL.NS": "Central Depository Services India Ltd",
        "BSE.NS": "BSE Ltd", "MAXHEALTH.NS": "Max Healthcare Institute Ltd",
        "LALPATHLAB.NS": "Dr Lal PathLabs Ltd", "MAZAGON.NS": "Mazagon Dock Shipbuilders Ltd",
        "BDL.NS": "Bharat Dynamics Ltd", "RVNL.NS": "Rail Vikas Nigam Ltd",
        "RAILTEL.NS": "RailTel Corporation of India Ltd", "SJVN.NS": "SJVN Ltd",
        "VBL.NS": "Varun Beverages Ltd", "LODHA.NS": "Macrotech Developers Ltd (Lodha)",
        "COCHINSHIP.NS": "Cochin Shipyard Ltd", "STARHEALTH.NS": "Star Health & Allied Insurance Co Ltd",
        "BANDHANBNK.NS": "Bandhan Bank Ltd", "HINDZINC.NS": "Hindustan Zinc Ltd",
        "BLUEDART.NS": "Blue Dart Express Ltd",
    })

    # ── Sector Classification ────────────────────────────────────────────────
    SECTORS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Banking & Finance": [
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
            "INDUSINDBK.NS", "BANKBARODA.NS", "CANBK.NS", "PNB.NS", "IDFCFIRSTB.NS",
            "FEDERALBNK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS",
            "MUTHOOTFIN.NS", "MANAPPURAM.NS", "SHRIRAMFIN.NS", "LICHSGFIN.NS",
            "SBICARD.NS", "YESBANK.NS", "BANDHANBNK.NS", "ANGELONE.NS",
            "CDSL.NS", "BSE.NS", "RECLTD.NS", "PFC.NS", "IRFC.NS",
        ],
        "IT & Technology": [
            "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
            "LTIM.NS", "PERSISTENT.NS", "KPITTECH.NS", "MPHASIS.NS", "LTTS.NS",
            "TATAELXSI.NS", "NAUKRI.NS", "OFSS.NS", "HAPPSTMNDS.NS", "Dixon.NS",
        ],
        "Pharma & Healthcare": [
            "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "LUPIN.NS",
            "AUROPHARMA.NS", "BIOCON.NS", "TORNTPHARM.NS", "GLENMARK.NS",
            "IPCALAB.NS", "LAURUSLABS.NS", "ZYDUSLIFE.NS", "APOLLOHOSP.NS",
            "FORTIS.NS", "MAXHEALTH.NS", "LALPATHLAB.NS",
        ],
        "Oil, Gas & Energy": [
            "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS",
            "HINDPETRO.NS", "PETRONET.NS", "ADANIGREEN.NS", "TATAPOWER.NS",
            "NTPC.NS", "POWERGRID.NS", "NHPC.NS", "JSWENERGY.NS", "SJVN.NS",
            "COALINDIA.NS", "ADANIPOWER.NS", "SUZLON.NS",
        ],
        "Automobile": [
            "TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS",
            "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "ESCORTS.NS", "MRF.NS",
            "BALKRISIND.NS", "BHARATFORG.NS", "MOTHERSON.NS",
        ],
        "Metals & Mining": [
            "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "NMDC.NS",
            "SAIL.NS", "JINDALSTEL.NS", "NATIONALUM.NS", "HINDZINC.NS",
        ],
        "FMCG & Consumer": [
            "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
            "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "TATACONSUM.NS", "VBL.NS",
            "DMART.NS", "JUBLFOOD.NS", "UBL.NS",
        ],
        "Cement & Building": [
            "ULTRACEMCO.NS", "GRASIM.NS", "AMBUJACEM.NS", "ACC.NS", "JKCEMENT.NS",
            "RAMCOCEM.NS", "PIDILITIND.NS", "ASTRAL.NS",
        ],
        "Real Estate": [
            "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "LODHA.NS", "PRESTIGE.NS",
            "BRIGADE.NS", "SOBHA.NS",
        ],
        "Defence & Aerospace": [
            "HAL.NS", "BEL.NS", "MAZAGON.NS", "BDL.NS", "COCHINSHIP.NS",
        ],
        "Insurance": [
            "SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "LICI.NS", "STARHEALTH.NS",
        ],
        "Telecom & Media": [
            "BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS", "ZEEL.NS", "SUNTV.NS",
        ],
        "Infrastructure & Capital Goods": [
            "LT.NS", "ADANIPORTS.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS",
            "POLYCAB.NS", "CGPOWER.NS", "CUMMINSIND.NS", "VOLTAS.NS",
            "RVNL.NS", "RAILTEL.NS", "IRCTC.NS",
        ],
        "Chemicals": [
            "SRF.NS", "PIIND.NS", "DEEPAKNTR.NS", "TATACHEM.NS", "UPL.NS",
            "CHAMBLFERT.NS", "COROMANDEL.NS",
        ],
        "New-Age / Fintech / E-Commerce": [
            "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "DELHIVERY.NS", "INDIGO.NS",
        ],
    })

    # Display names mapping (auto-generated from symbol)
    DISPLAY_NAMES: Dict[str, str] = field(default_factory=lambda: {
        "RELIANCE.NS": "RELIANCE", "TCS.NS": "TCS", "HDFCBANK.NS": "HDFCBANK",
        "INFY.NS": "INFY", "ICICIBANK.NS": "ICICIBANK", "SBIN.NS": "SBIN",
        "BHARTIARTL.NS": "BHARTIARTL", "ITC.NS": "ITC", "KOTAKBANK.NS": "KOTAKBANK",
        "LT.NS": "LT", "AXISBANK.NS": "AXISBANK", "WIPRO.NS": "WIPRO"
    })

    ALL_NSE_EQUITIES: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._load_nse_master()

    def _load_nse_master(self):
        """Load the complete official NSE master equity list (2,580+ listed companies)."""
        import os
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "EQUITY_L.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                valid_df = df[df['SERIES'].isin(['EQ', 'BE', 'SM', 'ST'])] if 'SERIES' in df.columns else df
                eqs = []
                for _, row in valid_df.iterrows():
                    sym_clean = str(row['SYMBOL']).strip()
                    if not sym_clean or sym_clean == 'nan':
                        continue
                    full_sym = f"{sym_clean}.NS"
                    eqs.append(full_sym)
                    comp_name = str(row.get('NAME OF COMPANY', '')).strip()
                    if comp_name and comp_name != 'nan':
                        if full_sym not in self.COMPANY_NAMES:
                            self.COMPANY_NAMES[full_sym] = comp_name
                self.ALL_NSE_EQUITIES = sorted(list(set(eqs)))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Could not load NSE master list: {e}")

    def get_all_symbols(self) -> List[str]:
        """Return de-duplicated sorted list of all NSE symbols (including full 2,580+ master list)."""
        all_syms = set()
        if self.ALL_NSE_EQUITIES:
            all_syms.update(self.ALL_NSE_EQUITIES)
        all_syms.update(self.NIFTY_50)
        all_syms.update(self.NIFTY_NEXT_50)
        all_syms.update(self.NIFTY_MIDCAP)
        all_syms.update(self.POPULAR_STOCKS)
        return sorted(all_syms)

    def get_display_name(self, symbol: str) -> str:
        """Return display name for a symbol (without .NS suffix)."""
        return symbol.replace(".NS", "")

    def get_company_name(self, symbol: str) -> str:
        """Return full company name, or display name if not found."""
        return self.COMPANY_NAMES.get(symbol, self.get_display_name(symbol))

    def get_sector(self, symbol: str) -> str:
        """Return the sector for a given symbol."""
        for sector, stocks in self.SECTORS.items():
            if symbol in stocks:
                return sector
        return "Other"

    def search_stocks(self, query: str) -> List[Dict[str, str]]:
        """
        Search stocks across every company listed on NSE by ticker or company name.
        Returns list of {symbol, display, company_name, sector}.
        """
        query = query.upper().strip()
        results = []
        all_syms = self.get_all_symbols()
        for sym in all_syms:
            display = self.get_display_name(sym)
            company = self.get_company_name(sym)
            if query in display.upper() or query in company.upper():
                results.append({
                    "symbol": sym,
                    "display": display,
                    "company_name": company,
                    "sector": self.get_sector(sym),
                })
        return results


@dataclass
class ModelConfig:
    # Chronological split to avoid data leakage
    TRAIN_START: str = "2018-01-01"
    TRAIN_END: str = "2022-12-31"
    VAL_START: str = "2023-01-01"
    VAL_END: str = "2024-12-31"
    TEST_START: str = "2025-01-01"
    TEST_END: str = "2026-09-17"
    
    # Feature windows
    LOOKBACK_DAYS: int = 60
    PREDICTION_HORIZON: int = 5
    RETURN_THRESHOLD: float = 0.02  # 2% for BUY signal
    
    # Technical indicators
    RSI_PERIOD: int = 14
    SMA_SHORT: int = 20
    SMA_LONG: int = 50
    EMA_SHORT: int = 12
    EMA_LONG: int = 26
    BB_PERIOD: int = 20
    ATR_PERIOD: int = 14
    VOLATILITY_WINDOW: int = 20

@dataclass
class AgentWeights:
    """Validated through backtesting - not arbitrary"""
    TECHNICAL: float = 0.25
    ML_PREDICTION: float = 0.25
    SENTIMENT: float = 0.15
    FUNDAMENTAL: float = 0.20
    MARKET_REGIME: float = 0.10
    RISK: float = 0.05
    
    def validate(self):
        total = self.TECHNICAL + self.ML_PREDICTION + self.SENTIMENT + self.FUNDAMENTAL + self.MARKET_REGIME + self.RISK
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"

@dataclass
class RiskConfig:
    MAX_POSITION_SIZE: float = 0.15  # 15% max per stock
    STOP_LOSS: float = 0.08  # 8%
    TAKE_PROFIT: float = 0.20  # 20%
    HIGH_VOLATILITY_THRESHOLD: float = 0.35
    ELEVATED_VIX_THRESHOLD: float = 20.0

# Global instances
STOCK_UNIVERSE = StockUniverse()
MODEL_CONFIG = ModelConfig()
AGENT_WEIGHTS = AgentWeights()
RISK_CONFIG = RiskConfig()
AGENT_WEIGHTS.validate()
