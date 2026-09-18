import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "Opytimizer",
  tagline: "Optimization library for Python",

  favicon: "img/logo.png",

  url: "https://recogna-lab.github.io",
  baseUrl: "/opytimizer/",

  organizationName: "recogna-lab",
  projectName: "opytimizer",

  trailingSlash: false,

  onBrokenLinks: "throw",

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },

        blog: false,

        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],
  themeConfig: {
    navbar: {
      title: 'Opytimizer',
      logo: {
        alt: 'Opytimizer Logo',
        src: 'img/logo.png', 
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar', 
          position: 'left',
          label: 'Documentation', 
        },
        {
          href: 'https://github.com/recogna-lab/opytimizer', 
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Community',
          items: [
            {
              label: 'Maintained by Recogna',
              href: 'https://recogna.tech/'
            }
          ],
        }
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Opytimizer Project. Built with Docusaurus.`
    }
  },
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
    },
  ],
};

export default config;