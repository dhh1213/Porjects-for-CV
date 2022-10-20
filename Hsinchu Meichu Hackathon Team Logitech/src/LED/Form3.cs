using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace WindowsFormsApp1
{
    public partial class Form3 : Form
    {
        public Form3()
        {
            InitializeComponent();
            textBox1.Text = Convert.ToString(Program.corret_n);
            textBox2.Text = Convert.ToString(Program.error_n);
            float acc = ((float)Program.corret_n / (Program.corret_n + Program.error_n)) * 100;
            textBox3.Text = Convert.ToString(acc);
            textBox4.Text = Convert.ToString((float)Program.corret_n / ((Program.m) * 60 + Program.s));
            textBox5.Text = Convert.ToString((float)Program.corret_n / (Program.lines.Length));
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            Program.corret_n = 0;
            Program.error_n = 0;
            this.Close();
        }

        private void label6_Click(object sender, EventArgs e)
        {

        }
    }
}
