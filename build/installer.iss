[Setup]
AppName=WhatsApp Excel
AppVersion=1.0
AppPublisher=Inocos
AppPublisherURL=https://inocos.pt
DefaultDirName={autopf}\WhatsAppExcel
DefaultGroupName=WhatsApp Excel
OutputDir=..\dist
OutputBaseFilename=WhatsAppExcel_Instalador
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "portuguese"; MessagesFile: "compiler:Languages\Portuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar ícone no Ambiente de Trabalho"; GroupDescription: "Opções:"

[Files]
Source: "..\dist\WhatsAppExcel.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\WhatsApp Excel";             Filename: "{app}\WhatsAppExcel.exe"
Name: "{group}\Desinstalar WhatsApp Excel"; Filename: "{uninstallexe}"
Name: "{autodesktop}\WhatsApp Excel";       Filename: "{app}\WhatsAppExcel.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\WhatsAppExcel.exe"; Description: "Abrir WhatsApp Excel agora"; Flags: nowait postinstall skipifsilent
