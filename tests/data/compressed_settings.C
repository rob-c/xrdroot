// Writes compressed-settings-6.40.root: one histogram per compression setting, each
// key compressed by ROOT 6.40 with the setting its name spells, for checking that
// xrdroot packs the same bytes. Run with: root -b -q compressed_settings.C
void compressed_settings()
{
   TFile f("compressed-settings-6.40.root", "RECREATE");
   gRandom->SetSeed(4357);
   for (int setting : {101, 106, 201, 207, 209, 404, 409, 501, 505, 509}) {
      f.SetCompressionSettings(setting);
      TH1F h(Form("h%d", setting), "a histogram of many bins", 6000, -4, 4);
      h.FillRandom("gaus", 20000);
      h.Write();
      TNamed small(Form("n%d", setting), "a small object, compressed as small blocks are");
      small.Write();
   }
   f.Close();
}
