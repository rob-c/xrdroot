// A small TGeometry of every old shape the reader knows, written to a file, and drawn.
void mkgeom() {
   TCanvas *c = new TCanvas("c", "c", 300, 300);
   TGeometry *g = new TGeometry("small", "every old shape");
   new TMaterial("mat1", "IRON", 55.85, 26, 7.87);
   new TRotMatrix("turn", "turn", 90, 0, 90, 90, 0, 0);
   new TRotMatrix("tilt", "tilt", 90, 30, 90, 120, 0, 0);
   new TBRIK("BOX", "BOX", "void", 100, 60, 40);
   new TTRD1("TRD1", "TRD1", "void", 20, 10, 20, 30);
   new TTRD2("TRD2", "TRD2", "void", 20, 10, 20, 10, 30);
   new TTRAP("TRAP", "TRAP", "void", 30, 0, 0, 20, 10, 15, 5, 20, 10, 15, 5);
   new TGTRA("GTRA", "GTRA", "void", 30, 0, 0, 20, 20, 10, 15, 5, 20, 10, 15, 5);
   new TTUBE("TUBE", "TUBE", "void", 10, 20, 30);
   new TTUBS("TUBS", "TUBS", "void", 10, 20, 30, 0, 90);
   new TCONE("CONE", "CONE", "void", 30, 0, 10, 0, 20);
   new TCONS("CONS", "CONS", "void", 30, 0, 10, 0, 20, 0, 180);
   TSPHE *sphe = new TSPHE("SPHE", "SPHE", "void", 0, 20, 0, 180, 0, 360);
   sphe->SetLineColor(kRed);
   TNode *top = new TNode("TOP", "TOP", "BOX");
   top->cd();
   TNode *a = new TNode("A", "A", "TRD1", -60, 0, 0);
   a->SetLineColor(kBlue);
   new TNode("B", "B", "TRD2", 60, 0, 0, "turn");
   new TNode("C", "C", "TRAP", 0, 50, 0, "tilt");
   new TNode("D", "D", "GTRA", 0, -50, 0);
   new TNode("E", "E", "TUBE", -60, 50, 0);
   new TNode("F", "F", "TUBS", 60, 50, 0);
   new TNode("G", "G", "CONE", -60, -50, 0);
   TNode *h = new TNode("H", "H", "CONS", 60, -50, 0);
   h->cd();
   new TNode("I", "I", "SPHE", 0, 0, 40);
   top->SetVisibility(-4);
   top->Draw();
   c->SaveAs("tgeometry-small-6.40.png");
   TFile f("tgeometry-small-6.40.root", "RECREATE");
   g->Write();
   f.Close();
}
