using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using LedCSharp;
using System.IO;

namespace WindowsFormsApp1
{
    static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        public static int book = 0;
        public static int timespan = 0;
        public static int corret_n = 0;
        public static int error_n = 0;
        public static int cur_string = 0; // 當前單字數
        public static int cur_n_in_str = 0; // current count in string
        public static int h, m, s;
        public static string lines = File.ReadAllText(@"..\..\..\..\Resource\T1.txt");
        public static List<string> splited_str = new List<string>();
        public static char ans = lines[corret_n];



        [STAThread]
        static void Main()
        {
            Application.SetHighDpiMode(HighDpiMode.SystemAware);
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());
        }
    }
}
