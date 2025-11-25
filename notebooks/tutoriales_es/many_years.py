from enahopy.loader import ENAHODataDownloader

# Inicializar downloader
downloader = ENAHODataDownloader(verbose=True)

# Descargar múltiples módulos en paralelo
data_multi = downloader.download(
    modules=['01', '02', '05', '34'],  # Hogar, Persona, Empleo, Sumaria
    years=['2024', '2023'],
    output_dir='./data',
    decompress=True,
    load_dta=True,
    parallel=True,
    max_workers=3
)

# Extraer datasets
df_hogar = data_multi[('2024', '01')]['enaho01-2024-100']
df_persona = data_multi[('2024', '02')]['enaho01-2024-200']
df_empleo = data_multi[('2024', '05')]['enaho01a-2024-500']
df_sumaria = data_multi[('2024', '34')]['sumaria-2024']

print(f"✓ Hogares: {len(df_hogar):,} registros")
print(f"✓ Personas: {len(df_persona):,} registros")